"""
Stage 1: connector alignment on image captions. The LLM is frozen (no LoRA); only the connector
trains, with the same step budget for every connector type, as in LLaVA's pretraining stage.

    python study/align.py --llm Qwen/Qwen3.5-4B --connector bottleneck --out runs_study/align/qwen35-4b_bottleneck
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import CaptionDataset, PromptBuilder
from src.study.vlm import StudyVLM

CAPTION_PROMPT = ["Describe this image briefly."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--connector", choices=["mlp", "qformer", "bottleneck"], required=True)
    parser.add_argument("--captions_dir", default="./data/tasks/captions")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1200, help="optimizer steps (equal for all connectors)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.llm)
    builder = PromptBuilder(tok, CAPTION_PROMPT, ["No", "Yes"])
    data = CaptionDataset(args.captions_dir, seed=args.seed)
    vision_dim = data.features.shape[-1]
    model = StudyVLM(args.llm, args.connector, builder.before_ids, vision_dim=vision_dim, dtype=args.dtype,
                     gradient_checkpointing=args.gradient_checkpointing, device=device, use_lora=False)
    params = list(model.connector.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.03 * args.steps), args.steps)
    scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" else None
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    losses, step, start = [], 0, time.time()
    bar = tqdm(total=args.steps, desc=f"align {args.connector}")
    while step < args.steps:
        for feats, caps in loader:
            b = {k: v.to(device) for k, v in builder.caption_batch(list(caps)).items()}
            with torch.autocast("cuda", dtype=model.compute_dtype, enabled=device == "cuda"):
                hidden, _, _ = model(feats.to(device), b["after_ids"], b["after_mask"])
                logits, target = model.logits_at(hidden, b["labels"])
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            bar.update(1)
            if step % 50 == 0:
                bar.set_postfix(loss=f"{sum(losses[-50:]) / 50:.3f}")
            if step >= args.steps:
                break

    torch.save(model.connector.state_dict(), os.path.join(args.out, "connector.pt"))
    with open(os.path.join(args.out, "align.json"), "w") as f:
        json.dump({"args": vars(args), "connector_params": model.connector_param_count(),
                   "minutes": (time.time() - start) / 60, "final_loss": sum(losses[-50:]) / min(50, len(losses)),
                   "loss_curve": losses[::10]}, f, indent=2)
    print(f"aligned {args.connector}: final caption loss {sum(losses[-50:]) / min(50, len(losses)):.3f}")


if __name__ == "__main__":
    main()
