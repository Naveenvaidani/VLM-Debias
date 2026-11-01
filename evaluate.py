"""
Evaluate trained runs on the Waterbirds test split and export paper metrics + figures to ./metrics/.

All numbers come from model predictions on held-out data. ID = majority groups
(bird on its usual background), OOD = minority groups (bird on the other background).
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.waterbirds import (
    GROUP_NAMES, MINORITY_GROUPS, SPLITS, CachedFeatureDataset, find_waterbirds_root, load_metadata,
)
from src.models.debiased_vlm import DebiasedVLM
from src.training.engine import evaluate, make_loader

# Categorical palette (fixed order) and text tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

RUN_LABELS = {
    "mlp_prefix": "MLP projector (baseline)",
    "bottleneck": "VLM-SHIELD bottleneck",
    "mlp_prefix_balanced": "MLP projector + balanced",
    "bottleneck_balanced": "VLM-SHIELD + balanced",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, categories, runs, values, ylabel, title):
    width = 0.8 / len(runs)
    x = np.arange(len(categories))
    for i, run in enumerate(runs):
        bars = ax.bar(x + (i - (len(runs) - 1) / 2) * width, values[run], width * 0.92,
                      color=SERIES[i], label=RUN_LABELS.get(run, run))
        for bar, v in zip(bars, values[run]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7.5, color=TEXT_2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.set_ylim(0, 108)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    style_axes(ax)


def attention_figure(attn, test_set, meta, root, path, n=6):
    """Overlay the answer token's cross-attention (14x14 patches) on minority-group test images."""
    picks = [i for i in range(len(attn)) if test_set.group[i] in MINORITY_GROUPS][:n]
    fig, axes = plt.subplots(2, len(picks), figsize=(2.4 * len(picks), 5.2), facecolor=SURFACE)
    for col, i in enumerate(picks):
        img = Image.open(os.path.join(root, meta["img_filename"][test_set.indices[i]])).convert("RGB").resize((224, 224))
        side = int(np.sqrt(attn.shape[-1]))
        heat = attn[i].reshape(side, side)
        heat = np.array(Image.fromarray((heat / heat.max() * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC))
        axes[0, col].imshow(img)
        axes[0, col].set_title(GROUP_NAMES[test_set.group[i]], fontsize=8, color=TEXT)
        axes[1, col].imshow(img)
        axes[1, col].imshow(heat, cmap="Blues", alpha=0.6)
        for r in range(2):
            axes[r, col].axis("off")
    fig.suptitle("Answer-token cross-attention over SigLIP patches (minority-group test images)", color=TEXT, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["mlp_prefix", "bottleneck", "mlp_prefix_balanced", "bottleneck_balanced"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--features", default="./data/siglip_patch_features.npy")
    parser.add_argument("--metrics_dir", default="./metrics")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--single", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = find_waterbirds_root(args.data_dir)
    meta = load_metadata(root)
    test_set = CachedFeatureDataset(args.features, meta, "test")
    test_loader = make_loader(test_set, args.batch_size, train=False)

    if args.single:
        # Evaluate one checkpoint and cache its test results (one process per model,
        # since a 4-bit model's GPU memory is not fully released within a process).
        run = args.single
        ckpt = torch.load(os.path.join("runs", run, "best.pt"), map_location="cpu")
        train_args = ckpt["args"]
        model = DebiasedVLM(llm_model_name_or_path=train_args["llm"], mode=train_args["mode"],
                            lora_rank=train_args["lora_rank"], lora_alpha=2.0 * train_args["lora_rank"], device=device,
                            dtype=train_args.get("dtype", "bf16"), quantize_4bit=bool(train_args.get("quantize_4bit", 1)),
                            vision_dim=test_set.features.shape[-1])
        missing = set(ckpt["state"]) - set(dict(model.named_parameters()))
        assert not missing, f"checkpoint keys not in model: {list(missing)[:5]}"
        model.load_state_dict(ckpt["state"], strict=False)
        res = evaluate(model, test_loader, device, keep_attention=len(test_set) if model.mode == "bottleneck" else 0)
        res["metrics"]["best_epoch"] = ckpt["epoch"]
        res["metrics"]["llm"] = train_args["llm"]
        torch.save(res, os.path.join("runs", run, "test_results.pt"))
        print(run, {k: round(v, 4) if isinstance(v, float) else v for k, v in res["metrics"].items()})
        return

    runs = [r for r in args.runs if os.path.exists(os.path.join("runs", r, "best.pt"))]
    results, histories = {}, {}
    for run in runs:
        cached = os.path.join("runs", run, "test_results.pt")
        if not os.path.exists(cached):
            subprocess.run([sys.executable, __file__, "--single", run, "--data_dir", args.data_dir,
                            "--features", args.features, "--batch_size", str(args.batch_size)], check=True)
        res = torch.load(cached, weights_only=False)
        results[run] = res
        with open(os.path.join("runs", run, "history.json")) as f:
            histories[run] = json.load(f)
        if "attention" in res:
            attention_figure(res["attention"], test_set, meta, root,
                             os.path.join(args.metrics_dir, f"attention_{run}.png"))

    # Tables
    rows = []
    for run in runs:
        m = results[run]["metrics"]
        rows.append({
            "Model": RUN_LABELS.get(run, run), "LLM": m["llm"], "Best epoch": m["best_epoch"],
            "Avg acc (%)": 100 * m["avg_acc"], "Worst-group acc (%)": 100 * m["worst_group_acc"],
            "Acc_ID (%)": 100 * m["acc_id"], "Acc_OOD (%)": 100 * m["acc_ood"],
            "Delta_OOD (pp)": 100 * m["delta_ood"], "RSR": m["rsr"],
            **{f"{g} (%)": 100 * m[f"acc_{g}"] for g in GROUP_NAMES},
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.metrics_dir, "benchmark_summary.csv"), index=False, float_format="%.2f")
    with open(os.path.join(args.metrics_dir, "benchmark_results.json"), "w") as f:
        json.dump({
            "dataset": "Waterbirds (test split, n=%d)" % len(test_set),
            "id_groups": "landbird/land, waterbird/water", "ood_groups": "landbird/water, waterbird/land",
            "runs": {run: results[run]["metrics"] for run in runs},
            "histories": histories,
        }, f, indent=2)
    with open(os.path.join(args.metrics_dir, "research_paper_metrics.md"), "w") as f:
        f.write(f"# Waterbirds test results (n={len(test_set)})\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n")

    # Figure 1: headline accuracies
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    cats = ["Average", "Worst-group", "ID (majority)", "OOD (minority)"]
    vals = {r: [100 * results[r]["metrics"][k] for k in ("avg_acc", "worst_group_acc", "acc_id", "acc_ood")] for r in runs}
    grouped_bars(ax, cats, runs, vals, "Accuracy (%)", "Waterbirds test accuracy")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.metrics_dir, "accuracy_summary.png"), dpi=200, facecolor=SURFACE)
    plt.close()

    # Figure 2: per-group accuracy
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    vals = {r: [100 * results[r]["metrics"][f"acc_{g}"] for g in GROUP_NAMES] for r in runs}
    grouped_bars(ax, GROUP_NAMES, runs, vals, "Accuracy (%)", "Per-group test accuracy (bird / background)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.metrics_dir, "per_group_accuracy.png"), dpi=200, facecolor=SURFACE)
    plt.close()

    # Figure 3: validation curves (two charts, one measure each)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for i, run in enumerate(runs):
        ep = [h["epoch"] for h in histories[run]]
        ax1.plot(ep, [100 * h["val_avg_acc"] for h in histories[run]], color=SERIES[i], linewidth=2, marker="o", markersize=4, label=RUN_LABELS.get(run, run))
        ax2.plot(ep, [100 * h["val_worst_group_acc"] for h in histories[run]], color=SERIES[i], linewidth=2, marker="o", markersize=4)
    for ax, title in ((ax1, "Validation average accuracy (%)"), (ax2, "Validation worst-group accuracy (%)")):
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Epoch", color=TEXT_2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        style_axes(ax)
    ax1.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.metrics_dir, "training_curves.png"), dpi=200, facecolor=SURFACE)
    plt.close()

    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Saved metrics and figures to {os.path.abspath(args.metrics_dir)}")


if __name__ == "__main__":
    main()
