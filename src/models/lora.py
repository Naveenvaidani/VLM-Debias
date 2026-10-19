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
            
        # Low-rank matrices A and B, created on the wrapped layer's device
        device = original_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features), device=device))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank), device=device))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform for A, zeros for B (starts as identity perturbation 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_linear(x)
        lora_term = (x.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return result + (lora_term * self.scaling).to(result.dtype)

def apply_lora_to_llm(model: nn.Module, rank: int = 16, lora_alpha: float = 32.0, target_modules=("q_proj", "k_proj", "v_proj")):
    """
    Freezes every parameter of the LLM, then applies trainable LoRA adapters to
    specific linear projections. Returns the number of adapted layers.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Collect targets first: replacing children while iterating named_modules() is unsafe
    targets = [
        (module, child_name, child)
        for module in model.modules()
        for child_name, child in module.named_children()
        if child_name in target_modules and isinstance(child, nn.Linear)
    ]
    for module, child_name, child in targets:
        setattr(module, child_name, LoRALinear(child, rank=rank, lora_alpha=lora_alpha))
    return len(targets)
