import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class DualGroundingLoss(nn.Module):
    """
    Dual Grounding Loss for VLM-SHIELD:
    L_total = L_NTP + lambda_ent * L_entropy_penalty + lambda_supp * L_shortcut_suppression
    
    1. L_NTP: Standard Next-Token Prediction Causal Cross-Entropy Loss on shifted logits.
    2. L_entropy_penalty: Inverse entropy regularization on object tokens to prevent 
                          single-patch attention collapse onto background noise.
    3. L_shortcut_suppression: Explicit penalty driving attention on known spurious shortcut
                                patches (watermarks, borders) towards zero:
                                L_supp = sum_{p in Shortcut_Mask} A_p
    """
    def __init__(
        self, 
        lambda_entropy: float = 0.1, 
        lambda_suppression: float = 0.5, 
        lambda_sparsity: Optional[float] = None,
        eps: float = 1e-8
    ):
        super().__init__()
        if lambda_sparsity is not None:
            lambda_entropy = lambda_sparsity
        self.lambda_entropy = lambda_entropy
        self.lambda_suppression = lambda_suppression
        self.eps = eps

    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        attention_maps: torch.Tensor,
        shortcut_patch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            attention_maps: [batch, seq_len, num_patches] or [batch, num_heads, seq_len, num_patches]
            shortcut_patch_mask: [batch, num_patches] binary mask (1 for shortcut patches, 0 elsewhere)
            
        Returns:
            total_loss: scalar tensor for backward()
            loss_dict: dictionary containing individual loss metrics
        """
        # 1. Next-Token Prediction Loss (NTP)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ntp = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # 2. Extract attention probabilities
        if attention_maps.ndim == 4:
            attn_probs = attention_maps.mean(dim=1) # [batch, seq_len, num_patches]
        else:
            attn_probs = attention_maps
            
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + self.eps)
        
        # 3. Spatial Entropy Regularization (Inverse Entropy to prevent collapse)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + self.eps), dim=-1)
        loss_entropy = torch.mean(1.0 / (entropy + self.eps))
        
        # 4. Explicit Shortcut Patch Suppression
        if shortcut_patch_mask is not None:
            mask_expanded = shortcut_patch_mask.unsqueeze(1).float()
            shortcut_leakage = (attn_probs * mask_expanded).sum(dim=-1)
            loss_shortcut_supp = torch.mean(shortcut_leakage)
        else:
            loss_shortcut_supp = torch.tensor(0.0, device=logits.device)
            
        total_loss = (
            loss_ntp 
            + (self.lambda_entropy * loss_entropy) 
            + (self.lambda_suppression * loss_shortcut_supp)
        )
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_ntp": loss_ntp.item(),
            "loss_entropy": loss_entropy.item(),
            "loss_sparsity": loss_entropy.item(),
            "loss_shortcut_supp": loss_shortcut_supp.item() if isinstance(loss_shortcut_supp, torch.Tensor) else loss_shortcut_supp,
            "mean_entropy": entropy.mean().item()
        }
        
        return total_loss, loss_dict

# Alias for backward compatibility across test suites
DebiasedVLMLoss = DualGroundingLoss
