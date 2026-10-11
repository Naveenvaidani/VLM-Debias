"""
Complete VLM-SHIELD implementation from scratch in pure PyTorch.
Combines SigLIP Vision Encoder (from scratch), Qwen2.5 Language Backbone (from scratch),
and the custom LocalFeatureBottleneck (Cross-Attention Layer).
"""
import torch
import torch.nn as nn
from typing import Tuple

from .siglip_scratch import SiglipVisionEncoderScratch
from .qwen_scratch import Qwen2ForCausalLMScratch
from .debiased_vlm import LocalFeatureBottleneck

class DebiasedVLMScratch(nn.Module):
    """
    Complete VLM-SHIELD architecture built entirely from scratch in PyTorch.
    No external pre-trained dependencies required at runtime.
    """
    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 1536,
        image_size: int = 224,
        patch_size: int = 16,
        vocab_size: int = 151936,
        num_vision_layers: int = 6,
        num_llm_layers: int = 4,
        freeze_vision_encoder: bool = True
    ):
        super().__init__()
        # 1. Vision Encoder from Scratch
        self.vision_encoder = SiglipVisionEncoderScratch(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=vision_dim,
            num_layers=num_vision_layers
        )
        
        # 2. Language Backbone from Scratch
        self.llm = Qwen2ForCausalLMScratch(
            vocab_size=vocab_size,
            hidden_size=llm_dim,
            num_layers=num_llm_layers
        )
        
        # 3. Local Feature Bottleneck Layer
        self.bottleneck = LocalFeatureBottleneck(
            vision_dim=vision_dim,
            llm_dim=llm_dim
        )
        
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> Tuple[object, torch.Tensor]:
        """
        Forward pass through Vision Encoder -> LocalFeatureBottleneck -> LLM Backbone.
        
        Args:
            pixel_values: [batch, 3, 224, 224]
            input_ids:    [batch, seq_len]
            
        Returns:
            outputs: Object containing LLM logits [batch, seq_len, vocab_size]
            attention_maps: Cross-attention maps over 196 spatial vision patches [batch, seq_len, 196]
        """
        # 1. Extract dense patch embeddings [B, 196, 768]
        vision_patches = self.vision_encoder(pixel_values)
        
        # 2. Get LLM token embeddings [B, seq_len, 1536]
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 3. Route through Local Feature Bottleneck
        fused_features, attention_maps = self.bottleneck(vision_patches, text_embeds)
        
        # 4. Generate next tokens based strictly on attended local features
        outputs = self.llm(inputs_embeds=fused_features)
        return outputs, attention_maps
