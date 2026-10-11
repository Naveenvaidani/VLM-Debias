import torch
import torch.nn as nn
import os
from typing import Tuple, Optional, Union
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig

class LocalFeatureBottleneck(nn.Module):
    """
    Local Feature Bottleneck (Cross-Attention Layer)
    
    Replaces standard MLP projection funnels. The LLM's text token embeddings
    act as queries, explicitly cross-attending to spatial vision patch representations (keys/values).
    If a vision patch contains a spurious shortcut (e.g. background watermark or texture),
    the cross-attention mechanism allows the model to assign zero weight to that patch.
    """
    def __init__(self, vision_dim: int = 768, llm_dim: int = 1536, num_heads: int = 8):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        
        # Cross-Attention: LLM queries the Vision Patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.query_proj = nn.Linear(llm_dim, llm_dim)
        self.kv_proj = nn.Linear(vision_dim, llm_dim)
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(self, vision_patches: torch.Tensor, text_queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vision_patches: [batch, num_patches, vision_dim] (e.g. [batch, 196, 768] for SigLIP base 224)
            text_queries:   [batch, seq_len, llm_dim]      (e.g. [batch, seq_len, 1536] for Qwen2.5-1.5B)
            
        Returns:
            fused_features: [batch, seq_len, llm_dim]
            attn_weights:   [batch, seq_len, num_patches] or [batch, num_heads, seq_len, num_patches]
        """
        target_dtype = self.query_proj.weight.dtype
        if vision_patches.dtype != target_dtype:
            vision_patches = vision_patches.to(target_dtype)
        if text_queries.dtype != target_dtype:
            text_queries = text_queries.to(target_dtype)

        kv = self.kv_proj(vision_patches)
        q = self.query_proj(text_queries)
        
        # The LLM explicitly selects which local patches matter for the current token
        attn_output, attn_weights = self.cross_attn(query=q, key=kv, value=kv)
        
        fused_features = self.layer_norm(attn_output + q)
        return fused_features, attn_weights


class DebiasedVLM(nn.Module):
    """
    Debiased Vision-Language Model incorporating SigLIP Vision Encoder, 
    Qwen2.5 Language Backbone, and LocalFeatureBottleneck.
    """
    def __init__(
        self,
        vision_model_name_or_path: str = "google/siglip-base-patch16-224",
        llm_model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        local_files_only: bool = False,
        freeze_vision_encoder: bool = True,
        load_pretrained: bool = True
    ):
        super().__init__()
        self.vision_model_name_or_path = vision_model_name_or_path
        self.llm_model_name_or_path = llm_model_name_or_path
        
        if load_pretrained:
            # Check local paths first if available
            local_siglip = os.path.join(".", "local_models", "siglip-base-patch16-224")
            local_qwen = os.path.join(".", "local_models", "Qwen2.5-1.5B-Instruct")
            
            if os.path.exists(local_siglip):
                vision_path = local_siglip
            else:
                vision_path = vision_model_name_or_path
                
            if os.path.exists(local_qwen):
                llm_path = local_qwen
            else:
                llm_path = llm_model_name_or_path

            self.vision_encoder = AutoModel.from_pretrained(
                vision_path, 
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            
            vision_dim = getattr(self.vision_encoder.config, "hidden_size", 768)
            llm_dim = getattr(self.llm.config, "hidden_size", 1536)
        else:
            # Mock initialisation for unit testing
            vision_dim = 768
            llm_dim = 1536
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, vision_dim, kernel_size=16, stride=16),
                nn.Flatten(2)
            )
            self.llm = nn.Module()
            self.llm.config = type('Config', (), {'hidden_size': llm_dim, 'vocab_size': 151936})()
            self.llm.embed_tokens = nn.Embedding(151936, llm_dim)
            self.llm.get_input_embeddings = lambda: self.llm.embed_tokens
            self.llm.lm_head = nn.Linear(llm_dim, 151936)
            self.llm.dtype = torch.float32
            
            def mock_llm_forward(inputs_embeds=None, **kwargs):
                logits = self.llm.lm_head(inputs_embeds)
                return type('Outputs', (), {'logits': logits})()
            self.llm.forward = mock_llm_forward

        self.bottleneck = LocalFeatureBottleneck(
            vision_dim=vision_dim,
            llm_dim=llm_dim
        )
        
        # Align bottleneck dtype with LLM dtype if loaded
        if hasattr(self.llm, "dtype") and self.llm.dtype is not None:
            self.bottleneck = self.bottleneck.to(self.llm.dtype)
        
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
    def extract_vision_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract dense spatial visual patch embeddings from SigLIP.
        """
        if hasattr(self.vision_encoder, "vision_model"):
            vision_outputs = self.vision_encoder.vision_model(pixel_values).last_hidden_state
        elif hasattr(self.vision_encoder, "last_hidden_state"):
            vision_outputs = self.vision_encoder(pixel_values).last_hidden_state
        elif isinstance(self.vision_encoder, nn.Sequential):
            x = self.vision_encoder(pixel_values)
            vision_outputs = x.transpose(1, 2)
        else:
            outputs = self.vision_encoder(pixel_values)
            vision_outputs = getattr(outputs, "last_hidden_state", outputs[0])
            
        return vision_outputs

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> Tuple[object, torch.Tensor]:
        """
        Forward pass through frozen vision encoder, local bottleneck, and language backbone.
        """
        vision_outputs = self.extract_vision_patches(pixel_values)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        target_dtype = self.bottleneck.query_proj.weight.dtype
        if vision_outputs.dtype != target_dtype:
            vision_outputs = vision_outputs.to(target_dtype)
        if text_embeds.dtype != target_dtype:
            text_embeds = text_embeds.to(target_dtype)

        fused_features, attention_maps = self.bottleneck(vision_outputs, text_embeds)
        
        llm_dtype = getattr(self.llm, "dtype", torch.float32)
        if fused_features.dtype != llm_dtype:
            fused_features = fused_features.to(llm_dtype)

        outputs = self.llm(inputs_embeds=fused_features)
        return outputs, attention_maps

    @torch.no_grad()
    def generate_with_attention(
        self, 
        pixel_values: torch.Tensor, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        current_ids = input_ids.clone()
        final_attn = None
        
        for _ in range(max_new_tokens):
            outputs, attn_maps = self.forward(pixel_values, current_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            final_attn = attn_maps
            
        return current_ids, final_attn
