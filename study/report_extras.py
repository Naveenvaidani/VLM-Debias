"""
Paper extras from saved results (no GPU):

  compute_table.csv      parameters, training minutes, seconds per step and peak memory per connector
  failure_analysis.csv   per-group error rates over seeds and the share of test images every seed gets wrong
  attention_examples.png Waterbirds minority-group test images with the connectors' attention over patches

    python study/report_extras.py --runs_dir runs_study/qwen35-4b --waterbirds_dir data/waterbirds --out results_study
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

LABELS = {"mlp": "MLP projector", "qformer": "Q-Former", "bottleneck": "Cross-attn, learned gate",
          "bottleneck_fixed": "Cross-attn, fixed gate"}
# seconds per training step, batch 16, one T4 (study/bench_speed.py, NF4, no checkpointing, MLP pooled to 49 tokens)
STEP = {"mlp": (4.041, 12.11), "qformer": (2.479, 10.11), "bottleneck": (1.732, 6.24), "bottleneck_fixed": (1.732, 6.24)}


def compute_table(runs_dir):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "s*", "results.json")):
        task, conn = path.split(os.sep)[-4:-2]
        if conn not in LABELS:
            continue
        r = json.load(open(path))
        rows.append({"connector": conn, "task": task, "params_M": r["connector_params"] / 1e6,
                     "train_minutes": r["train_minutes"], "epochs_run": len(r["history"])})
    df = pd.DataFrame(rows).groupby("connector").agg(params_M=("params_M", "first"),
                                                       train_minutes_mean=("train_minutes", "mean"),
                                                       epochs_mean=("epochs_run", "mean")).reset_index()
    df["sec_per_step_T4"] = df.connector.map(lambda c: STEP[c][0])
    df["peak_memory_GB"] = df.connector.map(lambda c: STEP[c][1])
    df["image_tokens"] = df.connector.map({"mlp": 49, "qformer": 32, "bottleneck": 0, "bottleneck_fixed": 0})
    return df.round(2)


def failure_analysis(runs_dir):
    rows = []
    for conn_dir in sorted(glob.glob(os.path.join(runs_dir, "*", "*"))):
        task, conn = conn_dir.split(os.sep)[-2:]
        runs = [np.load(p) for p in sorted(glob.glob(os.path.join(conn_dir, "s*", "arrays.npz")))]
        if conn not in LABELS or not runs:
            continue
        labels, groups = runs[0]["labels"], runs[0]["groups"]
        wrong = np.stack([r["wga_t0_preds"] != labels for r in runs])  # [seeds, N], first question phrasing
        for g in np.unique(groups):
            m = groups == g
            rows.append({"task": task, "connector": conn, "group": int(g), "n": int(m.sum()),
                         "error_rate_mean": float(wrong[:, m].mean()),
                         "wrong_in_every_seed": float(wrong[:, m].all(0).mean()),
                         "wrong_in_any_seed": float(wrong[:, m].any(0).mean())})
    return pd.DataFrame(rows).round(3)


def attention_examples(runs_dir, wb_dir, out_png, n=4, seed=0):
    meta = pd.read_csv(os.path.join(wb_dir, "metadata.csv"))
    test = meta[meta["split"] == 2].reset_index(drop=True)
    conns = [c for c in ("qformer", "bottleneck", "bottleneck_fixed")
             if os.path.exists(os.path.join(runs_dir, "waterbirds", c, f"s{seed}", "arrays.npz"))]
    atts = {c: np.load(os.path.join(runs_dir, "waterbirds", c, f"s{seed}", "arrays.npz"))["wga_attention"]
            for c in conns}
    groups = 2 * test["y"].to_numpy() + test["place"].to_numpy()
    rng = np.random.default_rng(0)
    picks = np.concatenate([rng.choice(np.where(groups == g)[0], n // 2, replace=False) for g in (1, 2)])
    fig, axes = plt.subplots(len(picks), len(conns) + 1, figsize=(2.3 * (len(conns) + 1), 2.3 * len(picks)))
    for i, k in enumerate(picks):
        img = Image.open(os.path.join(wb_dir, test.loc[k, "img_filename"])).convert("RGB").resize((224, 224))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(["landbird / land", "landbird on water", "waterbird on land", "waterbird / water"][
            groups[k]], fontsize=8)
        for j, c in enumerate(conns, 1):
            a = atts[c][k].astype(np.float32).reshape(14, 14)
            axes[i, j].imshow(img)
            axes[i, j].imshow(np.kron(a / a.max(), np.ones((16, 16))), cmap="magma", alpha=0.55)
            if i == 0:
                axes[i, j].set_title(LABELS[c], fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Connector attention on Waterbirds minority-group test images (seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--waterbirds_dir", default="data/waterbirds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ct = compute_table(args.runs_dir)
    ct.to_csv(os.path.join(args.out, "compute_table.csv"), index=False)
    print(ct.to_string(index=False), "\n")
    fa = failure_analysis(args.runs_dir)
    fa.to_csv(os.path.join(args.out, "failure_analysis.csv"), index=False)
    print(fa.to_string(index=False), "\n")
    if os.path.exists(os.path.join(args.waterbirds_dir, "metadata.csv")):
        attention_examples(args.runs_dir, args.waterbirds_dir, os.path.join(args.out, "attention_examples.png"))
        print("attention_examples.png written")


if __name__ == "__main__":
    main()
