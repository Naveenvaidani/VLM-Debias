import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DualGroundingLoss(nn.Module):
    """
    Dual Grounding Loss for VLM-SHIELD:
    L_total = L_task + lambda_ent * L_entropy + lambda_supp * L_shortcut_suppression

    1. L_task: cross-entropy. Either next-token prediction on shifted logits
               ([batch, seq_len, vocab] logits + [batch, seq_len] labels) or
               answer-token classification ([batch, num_classes] logits + [batch] labels).
    2. L_entropy: 1 - H(A) / log(num_patches), in [0, 1]. Bounded penalty on low
                  attention entropy to prevent single-patch attention collapse.
    3. L_shortcut_suppression: attention mass on known spurious shortcut patches:
                               L_supp = sum_{p in Shortcut_Mask} A_p
    """
    def __init__(
        self,
        lambda_entropy: float = 0.1,
        lambda_suppression: float = 0.5,
        eps: float = 1e-8
    ):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_suppression = lambda_suppression
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        shortcut_patch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [batch, seq_len, vocab_size] or [batch, num_classes]
            labels: [batch, seq_len] (-100 = ignore) or [batch]
            attention_maps: [batch, seq_len, num_patches] or [batch, num_heads, seq_len, num_patches]
            shortcut_patch_mask: [batch, num_patches] binary mask (1 for shortcut patches, 0 elsewhere)

        Returns:
            total_loss: scalar tensor for backward()
            loss_dict: dictionary containing individual loss metrics
        """
        # 1. Task loss
        if logits.ndim == 3:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_task = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100
            )
        else:
            loss_task = F.cross_entropy(logits.float(), labels)

        total_loss = loss_task
        loss_dict = {"loss_task": loss_task.item()}

        if attention_maps is not None:
            # 2. Attention probabilities over patches
            attn_probs = attention_maps.float()
            if attn_probs.ndim == 4:
                attn_probs = attn_probs.mean(dim=1)
            attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + self.eps)

            # 3. Normalised entropy penalty (bounded, unlike 1 / H)
            num_patches = attn_probs.size(-1)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + self.eps), dim=-1)
            loss_entropy = torch.mean(1.0 - entropy / math.log(num_patches))
            total_loss = total_loss + self.lambda_entropy * loss_entropy
            loss_dict["loss_entropy"] = loss_entropy.item()
            loss_dict["mean_entropy"] = entropy.mean().item()

            # 4. Explicit shortcut patch suppression
            if shortcut_patch_mask is not None:
                mask_expanded = shortcut_patch_mask.unsqueeze(1).float()
                loss_shortcut_supp = (attn_probs * mask_expanded).sum(dim=-1).mean()
                total_loss = total_loss + self.lambda_suppression * loss_shortcut_supp
                loss_dict["loss_shortcut_supp"] = loss_shortcut_supp.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


# Alias for backward compatibility across test suites
DebiasedVLMLoss = DualGroundingLoss
