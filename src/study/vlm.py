"""
StudyVLM: frozen vision features -> connector -> any Hugging Face LLM text decoder (+ LoRA).

Works with text-only causal LMs (Llama, Mistral, Phi) and with native VLM checkpoints
(Qwen3.5, Gemma 3, Idefics2, GLM-4.6V, Pixtral): for the latter the model's own vision tower
and projector are dropped, so every backbone sees images only through the studied connector.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.study.connectors import build_connector

NATIVE_VISION_PARTS = ("visual", "vision_tower", "vision_model", "multi_modal_projector", "connector",
                       "vision_embed_tokens", "mm_projector")
QWEN35_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                       "gate_proj", "up_proj", "down_proj"]


def _find_decoder(llm: nn.Module) -> nn.Module:
    """The module mapping inputs_embeds -> last_hidden_state (has a `layers` stack)."""
    candidates = []
    if hasattr(llm, "get_decoder"):
        try:
            candidates.append(llm.get_decoder())
        except Exception:
            pass
    inner = getattr(llm, "model", None)
    for name in ("language_model", "text_model"):
        candidates.append(getattr(inner, name, None))
    candidates.append(inner)
    for c in candidates:
        if isinstance(c, nn.Module) and hasattr(c, "layers"):
            return c
    raise ValueError(f"could not locate the text decoder of {type(llm).__name__}")


class StudyVLM(nn.Module):
    def __init__(
        self,
        llm_name: str,
        connector: str,
        before_ids: List[int],
        vision_dim: int = 768,
        dtype: str = "fp16",
        quantize_4bit: bool = True,
        lora_rank: int = 16,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
        use_lora: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
                                  BitsAndBytesConfig)

        self.compute_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        config = AutoConfig.from_pretrained(llm_name)
        is_vlm = hasattr(config, "vision_config")
        quant = None
        if quantize_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                llm_int8_skip_modules=["lm_head", "embed_tokens", *NATIVE_VISION_PARTS],
            )
        cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        llm = cls.from_pretrained(llm_name, quantization_config=quant, dtype=self.compute_dtype,
                                  device_map={"": device}, low_cpu_mem_usage=True)
        for owner in (llm, getattr(llm, "model", None)):
            for part in NATIVE_VISION_PARTS:
                if owner is not None and isinstance(getattr(owner, part, None), nn.Module):
                    delattr(owner, part)
        torch.cuda.empty_cache()

        decoder = _find_decoder(llm)
        self.embed = llm.get_input_embeddings()
        self.lm_head = llm.get_output_embeddings()
        for p in llm.parameters():
            p.requires_grad = False
        if gradient_checkpointing:
            decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if use_lora:
            targets = QWEN35_LORA_TARGETS if config.model_type.startswith("qwen3_5") else "all-linear"
            decoder = get_peft_model(decoder, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                                         lora_dropout=0.05, bias="none", target_modules=targets))
        self.decoder = decoder
        llm_dim = self.embed.weight.shape[1]
        self.llm_dim = llm_dim
        self.connector = build_connector(connector, vision_dim, llm_dim).to(device=device, dtype=torch.float32)
        self.register_buffer("before_ids", torch.tensor(before_ids, dtype=torch.long, device=device))
        self.device_name = device

    def trainable_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def connector_param_count(self) -> int:
        return sum(p.numel() for p in self.connector.parameters())

    def forward(self, patches: torch.Tensor, after_ids: torch.Tensor, after_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Returns hidden states for the `after` positions [B, T, D], the connector's attention over
        patches (or None) and the sequence offset of the first `after` token.
        """
        b = patches.shape[0]
        before = self.embed(self.before_ids).unsqueeze(0).expand(b, -1, -1)
        after = self.embed(after_ids)
        n_before = before.shape[1]
        ones = torch.ones((b, n_before), dtype=after_mask.dtype, device=after_mask.device)
        if self.connector.kind == "prefix":
            img, attn = self.connector(patches.float())
            embeds = torch.cat([before, img.to(after.dtype), after], dim=1)
            img_ones = torch.ones((b, img.shape[1]), dtype=after_mask.dtype, device=after_mask.device)
            mask = torch.cat([ones, img_ones, after_mask], dim=1)
            offset = n_before + img.shape[1]
        else:
            text = torch.cat([before, after], dim=1)
            embeds, attn = self.connector(patches.float(), text.float())
            mask = torch.cat([ones, after_mask], dim=1)
            offset = n_before
        hidden = self.decoder(inputs_embeds=embeds.to(self.compute_dtype), attention_mask=mask,
                              use_cache=False).last_hidden_state
        return hidden[:, offset:], attn, offset

    def logits_at(self, hidden_after: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-vocabulary logits at labelled positions only (keeps memory small for 248k vocabs)."""
        sel = labels != -100
        h = hidden_after[sel]
        return self.lm_head(h.to(self.lm_head.weight.dtype)).float(), labels[sel]
