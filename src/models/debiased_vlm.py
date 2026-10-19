import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple

from src.data.waterbirds import CLASS_NAMES

PROMPT = (
    "<|im_start|>user\nIs the bird in this image a landbird or a waterbird? "
    "Answer with one word.<|im_end|>\n<|im_start|>assistant\n"
)


class LocalFeatureBottleneck(nn.Module):
    """
    Local Feature Bottleneck (Cross-Attention Layer)

    Replaces standard MLP projection funnels. The LLM's text token embeddings
    act as queries, explicitly cross-attending to spatial vision patch representations (keys/values).
    If a vision patch contains a spurious shortcut (e.g. background watermark or texture),
    the cross-attention mechanism allows the model to assign near-zero weight to that patch.

    The attended visual context is added to the text embeddings through a tanh gate
    initialised at zero, so training starts from the unmodified pretrained LLM.
    """
    def __init__(self, vision_dim: int = 768, llm_dim: int = 1536, num_heads: int = 8):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.num_heads = num_heads

        self.vision_norm = nn.LayerNorm(vision_dim)
        # kdim/vdim let MultiheadAttention project vision patches itself (no extra kv_proj)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            kdim=vision_dim,
            vdim=vision_dim,
            batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, vision_patches: torch.Tensor, text_queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_patches: [batch, num_patches, vision_dim] (e.g. [batch, 196, 768] for SigLIP base 224)
            text_queries:   [batch, seq_len, llm_dim]

        Returns:
            fused_features: [batch, seq_len, llm_dim]
            attn_weights:   [batch, seq_len, num_patches] (averaged over heads)
        """
        target_dtype = self.gate.dtype
        kv = self.vision_norm(vision_patches.to(target_dtype))
        q = text_queries.to(target_dtype)

        attn_output, attn_weights = self.cross_attn(query=q, key=kv, value=kv, need_weights=True)
        fused_features = q + torch.tanh(self.gate) * attn_output
        return fused_features, attn_weights


class MLPPrefixProjector(nn.Module):
    """
    Baseline LLaVA-style projector: pooled vision patches -> 2-layer MLP -> prefix tokens.
    Patches are average-pooled 2x2 (196 -> 49 tokens) to keep the LLM sequence short.
    """
    def __init__(self, vision_dim: int = 768, llm_dim: int = 1536, pool: int = 2):
        super().__init__()
        self.pool = pool
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_patches: torch.Tensor) -> torch.Tensor:
        b, p, d = vision_patches.shape
        side = int(p ** 0.5)
        grid = vision_patches.transpose(1, 2).reshape(b, d, side, side)
        pooled = F.avg_pool2d(grid, self.pool).flatten(2).transpose(1, 2)
        return self.proj(pooled.to(self.proj[1].weight.dtype))


def _text_backbone(llm: nn.Module) -> nn.Module:
    """Return the decoder stack that maps inputs_embeds -> last_hidden_state."""
    inner = llm.model
    return getattr(inner, "language_model", inner)


class DebiasedVLM(nn.Module):
    """
    SigLIP patch features + Qwen language backbone + vision connector.

    mode="bottleneck": LocalFeatureBottleneck (text queries cross-attend to patches).
    mode="mlp_prefix": baseline MLP projector whose outputs are prepended as prefix tokens.

    The LLM is loaded in 4-bit (QLoRA) with LoRA adapters; its (tied) token embedding
    matrix is kept on CPU since only a fixed prompt is embedded. Classification reads
    the LLM's next-token logits restricted to the class answer tokens.
    """
    def __init__(
        self,
        llm_model_name_or_path: str = "Qwen/Qwen3.5-4B",
        mode: str = "bottleneck",
        vision_dim: int = 768,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        quantize_4bit: bool = True,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
        class_names: Sequence[str] = CLASS_NAMES,
    ):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        assert mode in ("bottleneck", "mlp_prefix")
        self.mode = mode
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)

        quant_config = None
        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["lm_head", "embed_tokens", "visual"],
            )
        llm = AutoModelForImageTextToText.from_pretrained(
            llm_model_name_or_path,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        # Drop the native Qwen vision tower: images enter only through our connector.
        if hasattr(llm.model, "visual"):
            del llm.model.visual

        backbone = _text_backbone(llm)
        embed = backbone.embed_tokens
        llm_dim = embed.weight.shape[1]

        # Answer-token rows of the (tied) LM head -> 2-way classifier
        answer_ids = [self.tokenizer(name, add_special_tokens=False).input_ids[0] for name in class_names]
        assert len(set(answer_ids)) == len(answer_ids), f"class answers share first token: {answer_ids}"
        self.register_buffer("answer_head", embed.weight[answer_ids].detach().clone().float().to(device))
        self.register_buffer("prompt_embeds", embed(
            self.tokenizer(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(embed.weight.device)
        ).detach().to(device, torch.bfloat16))

        # Keep only the text decoder. The 248k-vocab embedding (tied with lm_head) is
        # replaced by a 1-row placeholder, freeing ~1.3 GB of GPU memory.
        backbone.embed_tokens = nn.Embedding(1, llm_dim)
        del llm
        torch.cuda.empty_cache()

        # Manual k-bit prep: peft's prepare_model_for_kbit_training would upcast all
        # non-quantised weights to fp32.
        for param in backbone.parameters():
            param.requires_grad = False
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.llm = get_peft_model(backbone, lora_config)
        self.backbone = self.llm

        if mode == "bottleneck":
            self.connector = LocalFeatureBottleneck(vision_dim=vision_dim, llm_dim=llm_dim)
        else:
            self.connector = MLPPrefixProjector(vision_dim=vision_dim, llm_dim=llm_dim)
        self.connector.to(device=device, dtype=torch.float32)

    def trainable_state_dict(self):
        """Only the parameters that training changes: LoRA adapters + connector."""
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def forward(self, vision_patches: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            vision_patches: [batch, num_patches, vision_dim] cached SigLIP features
        Returns:
            logits: [batch, num_classes] answer-token logits
            attention_maps: [batch, seq_len, num_patches] (bottleneck mode) or None
        """
        b = vision_patches.shape[0]
        text = self.prompt_embeds.expand(b, -1, -1)
        attention_maps = None
        if self.mode == "bottleneck":
            embeds, attention_maps = self.connector(vision_patches.float(), text.float())
        else:
            embeds = torch.cat([self.connector(vision_patches.float()), text.float()], dim=1)

        hidden = self.backbone(inputs_embeds=embeds.to(torch.bfloat16), use_cache=False).last_hidden_state
        logits = hidden[:, -1].float() @ self.answer_head.T
        return logits, attention_maps
