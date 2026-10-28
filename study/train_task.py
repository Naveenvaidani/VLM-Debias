"""
Stage 2: train on one shortcut task (LoRA + connector, natural/ERM sampling), then evaluate on the
group-balanced test split with every question template.

Two checkpoints are kept: best validation worst-group accuracy (uses group labels, standard in the
literature) and best validation average accuracy (no group labels); both are evaluated on test.

    python study/train_task.py --llm Qwen/Qwen3.5-4B --connector bottleneck --task_dir data/tasks/cooccur \
        --align_ckpt runs_study/align/qwen35-4b_bottleneck/connector.pt --seed 0 --run_dir runs_study/...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import TaskDataset, TaskSpec, PromptBuilder
from src.study.engine import evaluate, train_epoch
from src.study.device import get_device, is_xla, torch_device
from src.study.vlm import StudyVLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--connector", choices=["mlp", "qformer", "bottleneck", "bottleneck_fixed"], required=True)
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--align_ckpt", default=None, help="stage-1 connector weights (recommended)")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument("--connector_lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "xla", "cpu"], default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mlp_pool", type=int, default=1, help="avg-pool MLP patch tokens (2 -> 196 to 49)")
    parser.add_argument("--no_4bit", action="store_true", help="fp16 weights instead of NF4 (faster if it fits)")
    parser.add_argument("--time_budget_min", type=float, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0, help="debug only")
    parser.add_argument("--freeze_llm", action="store_true", help="no LoRA: only the connector trains")
    parser.add_argument("--save_adapter", action="store_true",
                        help="save the selected (val WGA) LoRA + connector weights as adapter.pt for stage 3")
    parser.add_argument("--init_from", default=None, help="adapter.pt of a trained run: stage-3 connector retraining")
    parser.add_argument("--train_on", choices=["train", "val_balanced", "jtt"], default="train",
                        help="stage 3: val_balanced = retrain the connector on a group-balanced half of validation "
                             "(DFR-style; the other half selects); jtt = retrain on the training set with the "
                             "initial model's errors upweighted (no group labels)")
    parser.add_argument("--jtt_upweight", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="cap val/test sets (smoke tests only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    if device == "xla" and args.dtype == "fp16":
        args.dtype = "bf16"  # TPUs compute in bf16
    device = torch_device(device)
    os.makedirs(args.run_dir, exist_ok=True)
    spec = TaskSpec(args.task_dir)
    tok = AutoTokenizer.from_pretrained(args.llm)
    builder = PromptBuilder(tok, spec.questions, spec.answers)
    if is_xla(get_device(args.device)):
        builder.pad_to = max(len(a) for a in builder.after_ids) + 1

    train_set = TaskDataset(spec, "train", seed=args.seed)
    if args.max_train_samples:
        train_set.indices, train_set.y, train_set.group = (x[:args.max_train_samples] for x in
                                                            (train_set.indices, train_set.y, train_set.group))
    def cap(ds):
        """Evenly spaced subset, so every group stays represented (smoke tests only)."""
        if args.max_eval_samples and len(ds.indices) > args.max_eval_samples:
            keep = np.linspace(0, len(ds.indices) - 1, args.max_eval_samples).astype(int)
            ds.indices, ds.y, ds.group = (np.asarray(x)[keep] for x in (ds.indices, ds.y, ds.group))
        return ds

    val_set = cap(TaskDataset(spec, "val", template=0))
    vision_dim = train_set.features.shape[-1]

    def subset(ds, idx):
        ds.indices, ds.y, ds.group = (np.asarray(x)[idx] for x in (ds.indices, ds.y, ds.group))
        return ds

    stage3 = {}
    if args.train_on == "val_balanced":  # DFR-style: fit on a balanced half of val, select on the other half
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(val_set.indices))
        fit, sel = perm[: len(perm) // 2], perm[len(perm) // 2:]
        groups_fit = val_set.group[fit]
        n = np.bincount(groups_fit, minlength=4)[np.bincount(groups_fit, minlength=4) > 0].min()
        fit = np.concatenate([rng.choice(fit[groups_fit == g], n, replace=False) for g in np.unique(groups_fit)])
        train_set = subset(TaskDataset(spec, "val", seed=args.seed), fit)
        val_set = subset(val_set, sel)
        stage3 = {"mode": "val_balanced", "n_fit": int(len(fit)), "n_select": int(len(sel))}

    model = StudyVLM(args.llm, args.connector, builder.before_ids, vision_dim=vision_dim, dtype=args.dtype,
                     lora_rank=args.lora_rank, gradient_checkpointing=args.gradient_checkpointing, device=device,
                     use_lora=not args.freeze_llm,
                     mlp_pool=args.mlp_pool, quantize_4bit=not args.no_4bit)
    if args.align_ckpt:
        model.connector.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"))
    if args.init_from:  # stage 3: start from a trained run and retrain only the connector
        state = torch.load(args.init_from, map_location="cpu")
        unexpected = model.load_state_dict(state, strict=False).unexpected_keys
        assert not unexpected, f"adapter keys not in model: {unexpected[:3]}"
        print(f"stage 3: loaded {len(state)} tensors from {args.init_from}", flush=True)
        for p in model.decoder.parameters():
            p.requires_grad = False
    if args.train_on == "jtt":  # errors of the initial model on the training set, upweighted
        probe_set = subset(TaskDataset(spec, "train", template=0), np.arange(len(train_set.indices)))  # same rows
        probe = DataLoader(probe_set, batch_size=args.batch_size * 2, num_workers=2)
        res = evaluate(model, builder, probe, device, spec.group_names)
        errors = np.where(res["preds"] != res["labels"])[0]
        idx = np.concatenate([np.arange(len(train_set.indices))] + [errors] * (args.jtt_upweight - 1))
        train_set = subset(train_set, idx)
        stage3 = {"mode": "jtt", "n_errors": int(len(errors)), "upweight": args.jtt_upweight,
                  "error_groups": np.bincount(res["groups"][errors], minlength=4).tolist()}

    lora_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr}] if lora_params else []
    param_groups = groups + [{"params": list(model.connector.parameters()), "lr": args.connector_lr}]
    if str(device).startswith("cuda"):
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(param_groups, weight_decay=0.01)
    else:  # bitsandbytes optimizers are CUDA-only
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, num_workers=2)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" and str(device).startswith("cuda") else None

    best = {"wga": (-1.0, None, 0), "avg": (-1.0, None, 0)}  # rule -> (val score, state, epoch)
    history, stale, start = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, builder, train_loader, optimizer, scheduler, scaler, device)
        val = evaluate(model, builder, val_loader, device, spec.group_names)["metrics"]
        minutes = (time.time() - t0) / 60
        gate = getattr(model.connector, "gate", None)
        history.append({"epoch": epoch, "minutes": minutes, "train_loss": loss,
                        "gate_tanh": float(torch.tanh(gate.float()).item()) if gate is not None else None,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{os.path.basename(args.run_dir)}] epoch {epoch} ({minutes:.1f} min) loss {loss:.4f} | "
              f"val avg {val['avg_acc']:.4f} wga {val['worst_group_acc']:.4f}", flush=True)
        improved = False
        for rule, key in (("wga", "worst_group_acc"), ("avg", "avg_acc")):
            if val[key] > best[rule][0]:
                best[rule] = (val[key], model.trainable_state_dict(), epoch)
                improved |= rule == "wga"
        stale = 0 if improved else stale + 1
        elapsed = (time.time() - start) / 60
        if stale >= args.patience or (args.time_budget_min and elapsed + minutes > args.time_budget_min):
            break

    # Test: every template, for both model-selection rules
    results = {"args": vars(args), "stage3": stage3, "connector_params": model.connector_param_count(), "history": history,
               "train_minutes": (time.time() - start) / 60, "test": {}}
    arrays = {}
    for rule, (score, state, epoch) in best.items():
        model.load_state_dict(state, strict=False)
        if rule == "wga":  # connector weights of the selected checkpoint (small; LoRA weights are not kept)
            torch.save(model.connector.state_dict(), os.path.join(args.run_dir, "connector.pt"))
            if args.save_adapter:
                torch.save(state, os.path.join(args.run_dir, "adapter.pt"))
        per_template = {}
        for t in range(len(spec.questions)):
            loader = DataLoader(cap(TaskDataset(spec, "test", template=t)), batch_size=args.batch_size * 2, num_workers=2)
            res = evaluate(model, builder, loader, device, spec.group_names, keep_attention=(t == 0))
            per_template[t] = res["metrics"]
            arrays[f"{rule}_t{t}_preds"] = res["preds"]
            if t == 0:
                arrays["labels"], arrays["groups"] = res["labels"], res["groups"]
                if "attention" in res:
                    arrays[f"{rule}_attention"] = res["attention"]
        keys = per_template[0].keys()
        mean = {k: float(np.mean([per_template[t][k] for t in per_template])) for k in keys}
        results["test"][rule] = {"best_epoch": epoch, "val_score": score, "mean_over_templates": mean,
                                 "per_template": per_template}
        print(f"test [{rule}] epoch {epoch}: avg {mean['avg_acc']:.4f} wga {mean['worst_group_acc']:.4f} "
              f"held-out template wga {per_template[len(spec.questions) - 1]['worst_group_acc']:.4f}", flush=True)
    with open(os.path.join(args.run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.run_dir, "arrays.npz"), **arrays)


if __name__ == "__main__":
    main()
