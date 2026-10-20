import math
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.waterbirds import CachedFeatureDataset, group_metrics


def make_loader(dataset: CachedFeatureDataset, batch_size: int, train: bool, balanced: bool = False) -> DataLoader:
    if train and balanced:
        # Group-balanced sampling: each of the 4 (label, background) groups equally likely
        counts = np.bincount(dataset.group, minlength=4).astype(np.float64)
        weights = 1.0 / counts[dataset.group]
        sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, keep_attention: int = 0) -> Dict:
    """Run the model over a split; returns group metrics plus attention statistics."""
    model.eval()
    preds, labels, groups, entropies, kept = [], [], [], [], []
    for feats, y, g in tqdm(loader, desc="eval", leave=False):
        with torch.autocast("cuda", dtype=model.compute_dtype, enabled=device == "cuda"):
            logits, attn = model(feats.to(device, non_blocking=True))
        preds.append(logits.argmax(-1).cpu())
        labels.append(y)
        groups.append(g)
        if attn is not None:
            # attention of the final prompt token (the one that produces the answer)
            last = attn[:, -1].float()
            last = last / last.sum(-1, keepdim=True)
            entropies.append((-(last * torch.log(last + 1e-8)).sum(-1)).cpu())
            if len(kept) * loader.batch_size < keep_attention:
                kept.append(last.cpu())
    preds, labels, groups = (torch.cat(t).numpy() for t in (preds, labels, groups))
    out = group_metrics(preds, labels, groups)
    if entropies:
        ent = torch.cat(entropies)
        out["attn_entropy"] = float(ent.mean())
        out["attn_entropy_normalised"] = float(ent.mean() / math.log(attn.shape[-1]))
    result = {"metrics": out, "preds": preds, "labels": labels, "groups": groups}
    if kept:
        result["attention"] = torch.cat(kept)[:keep_attention].numpy()
    return result


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device: str, grad_clip: float = 1.0,
                    scaler=None) -> Dict[str, float]:
    """`scaler`: a torch.amp.GradScaler, required when training in fp16 (T4 GPUs)."""
    model.train()
    totals: Dict[str, float] = {}
    for feats, y, _ in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=model.compute_dtype, enabled=device == "cuda"):
            logits, attn = model(feats.to(device, non_blocking=True))
        loss, loss_dict = loss_fn(logits, y.to(device), attn)
        params = [p for p in model.parameters() if p.requires_grad]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v
    return {k: v / len(loader) for k, v in totals.items()}
