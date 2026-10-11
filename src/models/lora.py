"""
Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning of Qwen2.5 Language Backbone.
"""
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 16, lora_alpha: float = 32.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zeros for B (starts as identity perturbation 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_linear(x)
        lora_term = (x @ self.lora_A.T) @ self.lora_B.T
        return result + (lora_term * self.scaling)

def apply_lora_to_llm(model: nn.Module, rank: int = 16, lora_alpha: float = 32.0, target_modules=("q_proj", "k_proj", "v_proj")):
    """
    Applies LoRA adapters to specific linear projections in the LLM.
    """
    applied_count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in target_modules and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, lora_alpha=lora_alpha)
                setattr(module, child_name, lora_layer)
                applied_count += 1
    return applied_count
