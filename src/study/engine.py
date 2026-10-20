from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.study.data import PromptBuilder, group_metrics


def _to(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def step_loss(model, builder: PromptBuilder, feats, y, templates, device):
    b = _to(builder.answer_batch(templates.tolist(), y.tolist()), device)
    with torch.autocast("cuda", dtype=model.compute_dtype, enabled=device == "cuda"):
        hidden, attn, _ = model(feats.to(device, non_blocking=True), b["after_ids"], b["after_mask"])
        logits, target = model.logits_at(hidden, b["labels"])
    return F.cross_entropy(logits, target)


def train_epoch(model, builder, loader, optimizer, scheduler, scaler, device, grad_clip=1.0) -> float:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    total = 0.0
    for feats, y, _, t in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss = step_loss(model, builder, feats, y, t, device)
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
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, builder: PromptBuilder, loader, device, group_names: List[str],
             keep_attention: bool = False) -> Dict:
    """
    Restricted prediction = argmax over the class answer tokens; generative prediction = argmax over
    the full vocabulary (counts as correct only if it is exactly the right answer token).
    Attention = the connector's attention over patches at the answer-predicting position
    (bottleneck) or averaged over queries (Q-Former).
    """
    model.eval()
    answer_ids = torch.tensor(builder.answer_ids, device=device)
    preds, gen_correct, labels, groups, attn_keep = [], [], [], [], []
    for feats, y, g, t in tqdm(loader, desc="eval", leave=False):
        b = _to(builder.answer_batch(t.tolist(), y.tolist()), device)
        with torch.autocast("cuda", dtype=model.compute_dtype, enabled=device == "cuda"):
            hidden, attn, offset = model(feats.to(device, non_blocking=True), b["after_ids"], b["after_mask"])
            logits, target = model.logits_at(hidden, b["labels"])
        preds.append(logits[:, answer_ids].argmax(-1).cpu())
        gen_correct.append((logits.argmax(-1) == target).cpu())
        labels.append(y)
        groups.append(g)
        if keep_attention and attn is not None:
            if model.connector.kind == "fuse":
                pos = (b["labels"] != -100).float().argmax(1) + len(model.before_ids)
                a = attn[torch.arange(attn.shape[0], device=attn.device), pos]
            else:
                a = attn.mean(1)
            attn_keep.append(a.float().cpu().half())
    preds, labels, groups = (torch.cat(x).numpy() for x in (preds, labels, groups))
    metrics = group_metrics(preds, labels, groups, group_names)
    metrics["generative_acc"] = float(torch.cat(gen_correct).float().mean())
    out = {"metrics": metrics, "preds": preds, "labels": labels, "groups": groups}
    if attn_keep:
        out["attention"] = torch.cat(attn_keep).numpy()
    return out
