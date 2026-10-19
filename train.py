"""
Train VLM-SHIELD (LocalFeatureBottleneck) or the MLP-prefix baseline on Waterbirds.

Frozen SigLIP features are read from the cache built by scripts/cache_siglip_features.py.
The Qwen backbone is loaded in 4-bit with LoRA adapters (QLoRA). The checkpoint with the
best validation worst-group accuracy is kept, with early stopping and a wall-clock budget.
"""
import argparse
import json
import os
import time

import torch
from transformers import get_cosine_schedule_with_warmup

from src.data.waterbirds import CachedFeatureDataset, find_waterbirds_root, load_metadata
from src.models.debiased_vlm import DebiasedVLM
from src.training.engine import evaluate, make_loader, train_one_epoch
from src.training.loss import DualGroundingLoss


def main():
    parser = argparse.ArgumentParser(description="Train VLM-SHIELD with QLoRA on Waterbirds")
    parser.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--mode", choices=["bottleneck", "mlp_prefix"], default="bottleneck")
    parser.add_argument("--balanced", action="store_true", help="Group-balanced sampling")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience (epochs)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--connector_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_entropy", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--time_budget_min", type=float, default=0, help="Stop before exceeding this (0 = none)")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Debug: subsample training set")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--features", default="./data/siglip_patch_features.npy")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"{args.mode}{'_balanced' if args.balanced else ''}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    meta = load_metadata(find_waterbirds_root(args.data_dir))
    train_set = CachedFeatureDataset(args.features, meta, "train")
    val_set = CachedFeatureDataset(args.features, meta, "val")
    if args.max_train_samples:
        train_set.indices = train_set.indices[:args.max_train_samples]
        train_set.y, train_set.group = train_set.y[:args.max_train_samples], train_set.group[:args.max_train_samples]
    train_loader = make_loader(train_set, args.batch_size, train=True, balanced=args.balanced)
    val_loader = make_loader(val_set, args.batch_size * 2, train=False)

    model = DebiasedVLM(
        llm_model_name_or_path=args.llm, mode=args.mode, lora_rank=args.lora_rank,
        lora_alpha=2.0 * args.lora_rank, gradient_checkpointing=args.gradient_checkpointing, device=device,
    )
    lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    connector_params = list(model.connector.parameters())
    print(f"Trainable: LoRA {sum(p.numel() for p in lora_params):,} | connector {sum(p.numel() for p in connector_params):,}")

    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit([
        {"params": lora_params, "lr": args.lr},
        {"params": connector_params, "lr": args.connector_lr},
    ], weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    loss_fn = DualGroundingLoss(lambda_entropy=args.lambda_entropy)

    history, best_wga, stale = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val = evaluate(model, val_loader, device)["metrics"]
        epoch_min = (time.time() - t0) / 60
        history.append({"epoch": epoch, "minutes": epoch_min, **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{run_name}] epoch {epoch}/{args.epochs} ({epoch_min:.1f} min) loss {train_stats['loss_task']:.4f} | "
              f"val avg {val['avg_acc']:.4f} worst-group {val['worst_group_acc']:.4f}", flush=True)

        if val["worst_group_acc"] > best_wga:
            best_wga, stale = val["worst_group_acc"], 0
            torch.save({"args": vars(args), "epoch": epoch, "state": model.trainable_state_dict()},
                       os.path.join(run_dir, "best.pt"))
        else:
            stale += 1
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        elapsed_min = (time.time() - start) / 60
        if stale >= args.patience:
            print(f"Early stopping: no worst-group improvement for {stale} epochs")
            break
        if args.time_budget_min and elapsed_min + epoch_min > args.time_budget_min:
            print(f"Stopping: next epoch would exceed time budget ({args.time_budget_min} min)")
            break

    print(f"Best val worst-group acc {best_wga:.4f}; checkpoint in {run_dir}/best.pt")


if __name__ == "__main__":
    main()
