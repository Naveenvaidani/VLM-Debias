"""
Where do the attention-based connectors look? For the watermark task the shortcut sits at a known
place (bottom-right corner), so the share of attention on those patches measures how much the
connector reads the shortcut. Uses the attention maps saved by train_task.py (test set, template 0).

    python study/analyze_attention.py --runs_dir runs_study/qwen35-4b --out results_study
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

GRID = 14  # SigLIP-base/16 at 224 px


def corner_mask(rows=2, cols=8):
    """Patches covered by the watermark: bottom `rows` rows, right `cols` columns (render_watermark)."""
    m = np.zeros((GRID, GRID), bool)
    m[-rows:, -cols:] = True
    return m.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule", default="wga")
    args = parser.parse_args()
    mask = corner_mask()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "watermark_p*", "*", "s*", "arrays.npz"))):
        a = np.load(path)
        key = f"{args.rule}_attention"
        if key not in a:
            continue
        att = a[key].astype(np.float32)
        att = att / att.sum(1, keepdims=True)
        corner = att[:, mask].sum(1)
        groups, labels = a["groups"], a["labels"]
        marked = (groups % 2) == 1  # group = 2y + a, a = 1 means the watermark is present
        task, conn, seed = path.split(os.sep)[-4:-1]
        entropy = -(att * np.log(att + 1e-12)).sum(1) / np.log(att.shape[1])
        rows.append({"task": task, "connector": conn, "seed": int(seed[1:]),
                     "corner_share_marked": float(corner[marked].mean()),
                     "corner_share_unmarked": float(corner[~marked].mean()),
                     "uniform_share": float(mask.mean()),
                     "attention_entropy": float(entropy.mean())})
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "attention_watermark.csv"), index=False)
    summary = df.groupby("connector")[["corner_share_marked", "corner_share_unmarked", "attention_entropy"]].agg(
        ["mean", "std"]).round(3)
    print(df.round(3).to_string(index=False))
    print("\nmean ± sd over seeds (uniform corner share = %.3f):" % mask.mean())
    print(summary.to_string())


if __name__ == "__main__":
    main()
