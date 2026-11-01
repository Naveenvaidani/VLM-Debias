"""
Aggregate test results over seeds: mean +- std per (connector, sampling) configuration.
Reads runs/<name>/test_results.pt written by `evaluate.py --single <name>`.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

LABELS = {
    "mlp_prefix": "MLP projector",
    "bottleneck": "VLM-SHIELD bottleneck",
    "mlp_prefix_balanced": "MLP projector + balanced",
    "bottleneck_balanced": "VLM-SHIELD + balanced",
}
COLUMNS = [("avg_acc", "Avg acc (%)"), ("worst_group_acc", "Worst-group acc (%)"),
           ("acc_id", "Acc_ID (%)"), ("acc_ood", "Acc_OOD (%)"), ("rsr", "RSR")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="./runs")
    parser.add_argument("--metrics_dir", default="./metrics")
    args = parser.parse_args()

    per_config = {}
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "test_results.pt"))):
        name = os.path.basename(os.path.dirname(path))
        config = re.sub(r"_s\d+$", "", name)
        if config in LABELS:
            per_config.setdefault(config, []).append(torch.load(path, weights_only=False)["metrics"])
    if not per_config:
        raise SystemExit("no test_results.pt files found")

    rows = []
    for config in LABELS:
        if config not in per_config:
            continue
        runs = per_config[config]
        row = {"Model": LABELS[config], "Seeds": len(runs)}
        for key, label in COLUMNS:
            vals = np.array([m[key] for m in runs]) * (1 if key == "rsr" else 100)
            row[label] = f"{vals.mean():.2f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.2f}"
            row[f"_{key}"] = vals
        rows.append(row)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    os.makedirs(args.metrics_dir, exist_ok=True)
    df.to_csv(os.path.join(args.metrics_dir, "seed_summary.csv"), index=False)
    with open(os.path.join(args.metrics_dir, "seed_summary.md"), "w") as f:
        f.write("# Waterbirds test results, mean ± std over seeds\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor="#fcfcfb")
    keys = [("avg_acc", "Average"), ("worst_group_acc", "Worst-group")]
    width = 0.8 / len(rows)
    for i, row in enumerate(rows):
        means = [row[f"_{k}"].mean() for k, _ in keys]
        errs = [row[f"_{k}"].std(ddof=1) if len(row[f"_{k}"]) > 1 else 0 for k, _ in keys]
        x = np.arange(len(keys)) + (i - (len(rows) - 1) / 2) * width
        ax.bar(x, means, width * 0.92, yerr=errs, capsize=3, color=SERIES[i], label=row["Model"],
               error_kw={"ecolor": "#52514e", "elinewidth": 1})
        for xi, m in zip(x, means):
            ax.text(xi, m + 2.2, f"{m:.1f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([label for _, label in keys])
    ax.set_ylabel("Test accuracy (%)", color="#52514e")
    ax.set_ylim(0, 108)
    ax.set_title(f"Waterbirds test accuracy, mean ± std over {max(r['Seeds'] for r in rows)} seeds",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#e4e3df")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.metrics_dir, "seed_summary.png"), dpi=200, facecolor="#fcfcfb")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
