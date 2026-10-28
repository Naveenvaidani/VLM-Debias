"""
Statistical comparison of connectors from the saved test predictions (no retraining).

For every task and pair (A, B) of connectors:
  - difference in worst-group accuracy (mean over seeds, mean over question templates)
  - 95% bootstrap confidence interval: test images resampled within each group (stratified),
    the same resample applied to every run, 2,000 replicates
  - two-sided bootstrap p-value, and a paired t-test over seeds
  - hierarchical bootstrap: seeds resampled with replacement as well as test images, so the interval
    covers training variance too (the conservative, headline test)
  - Holm-Bonferroni correction over all comparisons within a task

    python study/significance.py --runs_dir runs_study/qwen35-4b --out results_study
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def wga(pred, labels, groups, idx):
    """Worst-group accuracy on the rows `idx` (a bootstrap resample)."""
    correct = (pred[idx] == labels[idx])
    g = groups[idx]
    return min(correct[g == k].mean() for k in np.unique(g))


def load_runs(conn_dir, rule):
    runs = []
    for path in sorted(glob.glob(os.path.join(conn_dir, "s*", "arrays.npz"))):
        a = np.load(path)
        keys = sorted(k for k in a if k.startswith(f"{rule}_t") and k.endswith("_preds"))
        runs.append({"seed": os.path.basename(os.path.dirname(path)), "preds": [a[k] for k in keys],
                     "labels": a["labels"], "groups": a["groups"]})
    return runs


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule", default="wga")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for task_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        task = os.path.basename(task_dir)
        conns = {os.path.basename(d): load_runs(d, args.rule) for d in sorted(glob.glob(os.path.join(task_dir, "*")))
                 if glob.glob(os.path.join(d, "s*", "arrays.npz"))}
        conns = {c: r for c, r in conns.items() if len(r) >= 2}
        if len(conns) < 2:
            continue
        labels, groups = next(iter(conns.values()))[0]["labels"], next(iter(conns.values()))[0]["groups"]
        by_group = [np.where(groups == k)[0] for k in np.unique(groups)]
        full = np.arange(len(labels))

        def per_seed(runs, idx):
            """Per-seed WGA (mean over question templates) on test rows `idx`."""
            return np.array([np.mean([wga(p, labels, groups, idx) for p in r["preds"]]) for r in runs])

        def score(runs, idx):
            return per_seed(runs, idx).mean()

        boots = [np.concatenate([rng.choice(g, len(g), replace=True) for g in by_group]) for _ in range(args.n_boot)]
        seed_boot = {c: np.array([per_seed(r, b) for b in boots]) for c, r in conns.items()}  # [n_boot, seeds]
        cache = {c: v.mean(1) for c, v in seed_boot.items()}
        point = {c: score(r, full) for c, r in conns.items()}
        task_rows = []
        for a, b in itertools.combinations(sorted(conns), 2):
            diff = cache[a] - cache[b]
            obs = point[a] - point[b]
            p_boot = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            seeds_a = {r["seed"]: np.mean([wga(p, labels, groups, full) for p in r["preds"]]) for r in conns[a]}
            seeds_b = {r["seed"]: np.mean([wga(p, labels, groups, full) for p in r["preds"]]) for r in conns[b]}
            common = sorted(set(seeds_a) & set(seeds_b))
            # hierarchical: resample seed indices (paired by position) inside every test-set replicate
            n = min(seed_boot[a].shape[1], seed_boot[b].shape[1])
            pick = rng.integers(0, n, size=(args.n_boot, n))
            h = np.take_along_axis(seed_boot[a][:, :n], pick, 1).mean(1) - \
                np.take_along_axis(seed_boot[b][:, :n], pick, 1).mean(1)
            p_hier = min(1.0, 2 * min((h <= 0).mean(), (h >= 0).mean()))
            t_p = stats.ttest_rel([seeds_a[s] for s in common], [seeds_b[s] for s in common]).pvalue \
                if len(common) >= 2 else np.nan
            task_rows.append({"task": task, "A": a, "B": b, "wga_A": point[a] * 100, "wga_B": point[b] * 100,
                              "diff_pp": obs * 100, "ci_low_pp": np.percentile(diff, 2.5) * 100,
                              "ci_high_pp": np.percentile(diff, 97.5) * 100, "p_bootstrap": p_boot,
                              "p_paired_t_seeds": t_p, "n_seeds": len(common),
                              "hier_ci_low_pp": np.percentile(h, 2.5) * 100, "hier_ci_high_pp": np.percentile(h, 97.5) * 100,
                              "p_hierarchical": p_hier})
        adj = holm(np.array([r["p_hierarchical"] for r in task_rows]))
        for r, p in zip(task_rows, adj):
            r["p_hier_holm"] = p
            r["significant_0.05"] = bool(p < 0.05)
        rows += task_rows
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, f"significance_{args.rule}.csv"), index=False)
    show = df.round({"wga_A": 1, "wga_B": 1, "diff_pp": 1, "ci_low_pp": 1, "ci_high_pp": 1, "p_bootstrap": 4,
                     "p_paired_t_seeds": 3, "hier_ci_low_pp": 1, "hier_ci_high_pp": 1, "p_hierarchical": 4,
                     "p_hier_holm": 4})
    with open(os.path.join(args.out, f"significance_{args.rule}.md"), "w") as f:
        f.write(f"# Pairwise connector comparisons (test WGA, {args.rule} selection)\n\n"
                f"Difference A - B in points; 95% CI from {args.n_boot} stratified bootstrap resamples of the test "
                "set (CI) and of seeds plus test images (hierarchical); p-values two-sided; significance uses the "
                "hierarchical p-value, Holm-corrected within each task.\n\n")
        f.write(show.to_markdown(index=False))
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
