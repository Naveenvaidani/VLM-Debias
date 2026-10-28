"""
Reference points on the same frozen SigLIP features the VLMs see (no LLM):

  erm_probe   logistic regression on the shortcut-biased training set
  dfr         Deep Feature Reweighting (Kirichenko et al., 2023): logistic regression retrained on a
              group-balanced half of the validation set; the other half picks the regulariser
  oracle_gb   logistic regression on a group-balanced subsample of the training set

Features are mean-pooled over patches. Test metrics use the same group_metrics as the VLM runs.

    python study/linear_baselines.py --tasks_dir data/tasks --tasks waterbirds,watermark_p0.95 --out results_study
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import TaskSpec, group_metrics

CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def pooled(spec, pool="mean", chunk=512):
    """mean: average over patches (768-d). grid7: 2x2-average-pooled patch grid kept spatially (49 x 768),
    the same tokens the MLP connector sees; meanmax: mean and max over patches concatenated."""
    feats = np.load(spec.features_path, mmap_mode="r")

    def one(x):
        if pool == "mean":
            return x.mean(1)
        if pool == "meanmax":
            return np.concatenate([x.mean(1), x.max(1)], 1)
        side = int(round(x.shape[1] ** 0.5))
        g = x.reshape(len(x), side // 2, 2, side // 2, 2, -1).mean((2, 4))
        return g.reshape(len(x), -1)

    return np.concatenate([one(np.asarray(feats[i:i + chunk], dtype=np.float32)) for i in range(0, len(feats), chunk)])


def balanced_idx(groups, rng, n=None):
    counts = np.bincount(groups, minlength=4)
    n = n or counts[counts > 0].min()
    return np.concatenate([rng.choice(np.where(groups == g)[0], n, replace=False) for g in range(4) if counts[g]])


def fit(x, y, c, seed):
    return LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", default="data/tasks")
    parser.add_argument("--tasks", default="waterbirds,watermark_p0.95")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pools", default="mean", help="comma list of mean,meanmax,grid7")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for task, pool in [(t, p) for t in args.tasks.split(",") for p in args.pools.split(",")]:
        spec = TaskSpec(os.path.join(args.tasks_dir, task))
        x = pooled(spec, pool)
        meta = spec.meta
        y = meta["y"].to_numpy()
        g = (2 * meta["y"] + meta["a"]).to_numpy()
        split = {s: np.where(meta["split"] == s)[0] for s in ("train", "val", "test")}
        scaler = StandardScaler().fit(x[split["train"]])
        x = scaler.transform(x)
        te = split["test"]
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            va = rng.permutation(split["val"])
            va_fit, va_sel = va[: len(va) // 2], va[len(va) // 2:]

            def best_c(train_idx):
                scores = {c: group_metrics(fit(x[train_idx], y[train_idx], c, seed).predict(x[va_sel]), y[va_sel],
                                           g[va_sel], spec.group_names)["worst_group_acc"] for c in CS}
                return max(scores, key=scores.get)

            tr = split["train"]
            setups = {"erm_probe": tr,
                      "oracle_gb": tr[balanced_idx(g[tr], rng)],
                      "dfr": va_fit[balanced_idx(g[va_fit], rng)]}
            for name, idx in setups.items():
                c = best_c(idx)
                pred = fit(x[idx], y[idx], c, seed).predict(x[te])
                m = group_metrics(pred, y[te], g[te], spec.group_names)
                rows.append({"task": task, "features": pool, "method": name, "seed": seed, "C": c, "n_train": len(idx),
                             **{k: m[k] for k in ("avg_acc", "worst_group_acc", "acc_id", "acc_ood")}})
                print(f"{task} {pool} {name} s{seed}: wga {m['worst_group_acc']:.3f} avg {m['avg_acc']:.3f} (C={c}, n={len(idx)})",
                      flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "linear_baselines.csv"), index=False)
    summary = df.groupby(["task", "features", "method"])[["worst_group_acc", "avg_acc"]].agg(["mean", "std"]).mul(100).round(1)
    print(summary.to_string())
    with open(os.path.join(args.out, "linear_baselines.md"), "w") as f:
        f.write("# Linear baselines on frozen SigLIP features (mean-pooled), test %, mean ± s.d. over seeds\n\n")
        f.write(summary.to_markdown())


if __name__ == "__main__":
    main()
