"""
Seed ensemble: majority vote of the 3 seeds' test predictions (saved in arrays.npz), per task and
connector, for both checkpoint-selection rules and every question template. No retraining.

    python study/seed_ensemble.py --runs_dir runs_study/qwen35-4b --out results_study
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import group_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for conn_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*", "*"))):
        task, conn = conn_dir.split(os.sep)[-2:]
        runs = sorted(glob.glob(os.path.join(conn_dir, "s*", "arrays.npz")))
        if len(runs) < 2:
            continue
        arrays = [np.load(p) for p in runs]
        labels, groups = arrays[0]["labels"], arrays[0]["groups"]
        names = [f"g{g}" for g in range(4)]
        for rule in ("wga", "avg"):
            keys = sorted(k for k in arrays[0] if k.startswith(f"{rule}_t") and k.endswith("_preds"))
            per_t = []
            for k in keys:
                votes = np.stack([a[k] for a in arrays])  # [seeds, N]
                pred = (votes.mean(0) > 0.5).astype(int)  # majority of binary predictions
                per_t.append(group_metrics(pred, labels, groups, names))
            singles = [json.load(open(p.replace("arrays.npz", "results.json")))["test"][rule]["mean_over_templates"]
                       for p in runs]
            rows.append({"task": task, "connector": conn, "rule": rule, "seeds": len(runs),
                         "ensemble_wga": float(np.mean([m["worst_group_acc"] for m in per_t])),
                         "ensemble_avg": float(np.mean([m["avg_acc"] for m in per_t])),
                         "single_wga_mean": float(np.mean([s["worst_group_acc"] for s in singles])),
                         "single_avg_mean": float(np.mean([s["avg_acc"] for s in singles]))})
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "seed_ensemble.csv"), index=False)
    show = df[df.rule == "wga"].copy()
    for c in ("ensemble_wga", "single_wga_mean", "ensemble_avg", "single_avg_mean"):
        show[c] = (show[c] * 100).round(1)
    show["gain_wga"] = (show.ensemble_wga - show.single_wga_mean).round(1)
    print(show.drop(columns=["rule"]).to_string(index=False))


if __name__ == "__main__":
    main()
