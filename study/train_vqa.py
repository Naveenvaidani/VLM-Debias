"""
Stage 2 on VQA-CP v2 (open-ended answers, natural images, changing priors): LoRA + connector trained on the
VQA-CP train subset, checkpoint chosen on in-distribution validation VQA accuracy (no OOD data, no group labels),
then evaluated by free-form generation on an in-distribution test set (same priors as training) and on the
VQA-CP test split (shifted priors), with and without the visual pathway (inference-time ablation).

    python study/train_vqa.py --llm Qwen/Qwen3.5-4B --connector bottleneck --task_dir data/tasks/vqacp \
        --align_ckpt runs/align/bottleneck/connector.pt --seed 0 --run_dir runs/vqacp/bottleneck/s0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.connectors import CONNECTOR_NAMES
from src.study.device import autocast, get_device, torch_device
from src.study.engine import clip_and_log, learned_gates
from src.study.vlm import StudyVLM
from src.study.vqa import VQABuilder, VQADataset, VQASpec, generate, vqa_metrics


def run_eval(model, builder, spec, split, device, batch_size, max_n=0, ablate=False):
    model.eval()
    model.ablate_visual = ablate
    model.inject_log = []
    ds = VQADataset(spec, split)
    if max_n and len(ds) > max_n:
        ds.rows = ds.rows[np.linspace(0, len(ds.rows) - 1, max_n).astype(int)]
    rows, preds = [], []
    for feats, r in tqdm(DataLoader(ds, batch_size=batch_size, num_workers=2), desc=f"gen {split}", leave=False):
        qs = spec.questions.loc[r.numpy(), "question"].tolist()
        with autocast(device, model.compute_dtype):
            preds += generate(model, builder, feats.to(device), qs, device)
        rows.append(r.numpy())
    model.ablate_visual = False
    rows = np.concatenate(rows)
    m = vqa_metrics(spec, rows, preds)
    if getattr(model, "inject_log", None) and not ablate:
        m["inject_ratio"] = float(np.mean(model.inject_log))
    return m, rows, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", required=True)
    p.add_argument("--connector", choices=CONNECTOR_NAMES, required=True)
    p.add_argument("--task_dir", required=True)
    p.add_argument("--align_ckpt", default=None)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--connector_lr", type=float, default=2e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--device", choices=["auto", "cuda", "xla", "cpu"], default="auto")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mlp_pool", type=int, default=1)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--time_budget_min", type=float, default=0)
    p.add_argument("--max_train_samples", type=int, default=0, help="debug only")
    p.add_argument("--max_eval_samples", type=int, default=0, help="cap val/test sets (smoke tests only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch_device(get_device(args.device))
    os.makedirs(args.run_dir, exist_ok=True)
    spec = VQASpec(args.task_dir)
    tok = AutoTokenizer.from_pretrained(args.llm)
    builder = VQABuilder(tok)
    train_set = VQADataset(spec, "train")
    if args.max_train_samples:
        train_set.rows = train_set.rows[:args.max_train_samples]
    model = StudyVLM(args.llm, args.connector, builder.before_ids, vision_dim=train_set.features.shape[-1],
                     dtype=args.dtype, lora_rank=args.lora_rank, gradient_checkpointing=args.gradient_checkpointing,
                     device=device, mlp_pool=args.mlp_pool, quantize_4bit=not args.no_4bit)
    if args.align_ckpt:
        model.connector.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"))
    lora = [q for q in model.decoder.parameters() if q.requires_grad]
    groups = ([{"params": lora, "lr": args.lr}] if lora else []) + \
             [{"params": list(model.connector.parameters()), "lr": args.connector_lr}]
    if str(device).startswith("cuda"):
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(groups, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=2)
    total = max(1, args.epochs * len(loader))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" and str(device).startswith("cuda") else None
    params = [q for q in model.parameters() if q.requires_grad]

    best, history, stale, start = (-1.0, None, 0), [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for feats, r in tqdm(loader, desc="train", leave=False):
            q = spec.questions.loc[r.numpy()]
            b = {k: v.to(device) for k, v in builder.qa_batch(q["question"].tolist(), q["answer"].tolist()).items()}
            opt.zero_grad(set_to_none=True)
            with autocast(device, model.compute_dtype):
                hidden, _, _ = model(feats.to(device, non_blocking=True), b["after_ids"], b["after_mask"])
                logits, target = model.logits_at(hidden, b["labels"])
                loss = F.cross_entropy(logits, target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clip_and_log(model, params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                clip_and_log(model, params, 1.0)
                opt.step()
            sched.step()
            losses.append(loss.item())
        val, _, _ = run_eval(model, builder, spec, "val", device, args.batch_size * 2, args.max_eval_samples)
        gate = getattr(model.connector, "gate", None)
        minutes = (time.time() - t0) / 60
        history.append({"epoch": epoch, "minutes": minutes, "train_loss": float(np.mean(losses)),
                        "gate_tanh": float(torch.tanh(gate.float()).item()) if gate is not None else None,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{os.path.basename(args.run_dir)}] epoch {epoch} ({minutes:.1f} min) loss {np.mean(losses):.4f} "
              f"| val vqa_acc {val['vqa_acc']:.4f}", flush=True)
        if val["vqa_acc"] > best[0]:
            best, stale = (val["vqa_acc"], model.trainable_state_dict(), epoch), 0
        else:
            stale += 1
        elapsed = (time.time() - start) / 60
        if stale >= args.patience or (args.time_budget_min and elapsed + 2 * minutes > args.time_budget_min):
            break

    model.load_state_dict(best[1], strict=False)
    results = {"args": vars(args), "connector_params": model.connector_param_count(), "history": history,
               "best_epoch": best[2], "val_vqa_acc": best[0], "train_minutes": (time.time() - start) / 60,
               "test": {}}
    preds_out = {}
    for split in ("test_id", "test_ood"):
        for ablate in (False, True):
            m, rows, preds = run_eval(model, builder, spec, split, device, args.batch_size * 2,
                                      args.max_eval_samples, ablate=ablate)
            results["test"][f"{split}{'_ablated' if ablate else ''}"] = m
            if not ablate:
                preds_out[split] = {"question_id": spec.questions.loc[rows, "question_id"].tolist(), "pred": preds}
    json.dump(results, open(os.path.join(args.run_dir, "results.json"), "w"), indent=1)
    json.dump(preds_out, open(os.path.join(args.run_dir, "predictions.json"), "w"))
    t = results["test"]
    print(f"[{os.path.basename(args.run_dir)}] test_id {t['test_id']['vqa_acc']:.4f} test_ood "
          f"{t['test_ood']['vqa_acc']:.4f} (ablated {t['test_ood_ablated']['vqa_acc']:.4f})", flush=True)


if __name__ == "__main__":
    main()
