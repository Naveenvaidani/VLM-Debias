"""
Aggregate study runs: mean ± std over seeds per (task, connector), paired connector differences,
and bootstrap 95% CIs on worst-group accuracy (resampling test examples within each group).

    python study/aggregate.py --runs_dir runs_study/qwen35-4b --out runs_study/qwen35-4b/summary
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

METRICS = [("avg_acc", "Avg"), ("worst_group_acc", "WGA"), ("acc_id", "ID"), ("acc_ood", "OOD"),
           ("rsr", "RSR"), ("balanced_acc", "Bal."), ("generative_acc", "Gen.")]
SERIES = {"mlp": "#2a78d6", "qformer": "#eb6834", "bottleneck": "#1baf7a"}
LABELS = {"mlp": "MLP projector", "qformer": "Q-Former", "bottleneck": "Cross-attn bottleneck"}


def bootstrap_wga(preds, labels, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    correct = preds == labels
    idx = [np.nonzero(groups == g)[0] for g in range(4)]
    stats = []
    for _ in range(n):
        accs = [correct[rng.choice(ix, len(ix))].mean() for ix in idx if len(ix)]
        stats.append(min(accs))
    return np.percentile(stats, [2.5, 97.5])


def load(runs_dir):
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "*", "*", "s*", "results.json")):
        seed_dir = os.path.dirname(path)
        task, connector = seed_dir.split(os.sep)[-3:-1]
        r = json.load(open(path))
        arrays = np.load(os.path.join(seed_dir, "arrays.npz"))
        for rule, res in r["test"].items():
            m = res["mean_over_templates"]
            held = res["per_template"][str(max(int(k) for k in res["per_template"]))]
            lo, hi = bootstrap_wga(arrays[f"{rule}_t0_preds"], arrays["labels"], arrays["groups"])
            rows.append({"task": task, "connector": connector, "seed": int(seed_dir[-1]), "rule": rule,
                         **{k: m[k] for k, _ in METRICS}, "heldout_wga": held["worst_group_acc"],
                         "wga_ci_lo": lo, "wga_ci_hi": hi, "best_epoch": res["best_epoch"],
                         "connector_params": r["connector_params"], "train_minutes": r["train_minutes"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.runs_dir)
    if df.empty:
        raise SystemExit("no results yet")
    df.to_csv(os.path.join(args.out, "all_runs.csv"), index=False)

    lines = ["# Connector study results\n"]
    for rule, title in (("wga", "selected by validation worst-group accuracy"),
                        ("avg", "selected by validation average accuracy (no group labels)")):
        d = df[df.rule == rule]
        agg = d.groupby(["task", "connector"]).agg(
            seeds=("seed", "nunique"), **{f"{lab}": (k, "mean") for k, lab in METRICS},
            **{f"{lab}_sd": (k, "std") for k, lab in METRICS}, heldout_WGA=("heldout_wga", "mean")).reset_index()
        table = agg[["task", "connector", "seeds"]].copy()
        for _, lab in METRICS:
            scale = 1 if lab == "RSR" else 100
            table[lab] = [f"{m * scale:.1f} ± {0 if np.isnan(s) else s * scale:.1f}"
                          for m, s in zip(agg[lab], agg[f"{lab}_sd"])]
        table["Held-out template WGA"] = (agg["heldout_WGA"] * 100).round(1)
        lines += [f"\n## Test, {title}\n", table.to_markdown(index=False), "\n"]

        # paired differences vs the MLP baseline (same task, same seed)
        piv = d.pivot_table(index=["task", "seed"], columns="connector", values="worst_group_acc")
        diffs = []
        for c in ("qformer", "bottleneck"):
            if c in piv and "mlp" in piv:
                delta = (piv[c] - piv["mlp"]).dropna()
                for task, v in delta.groupby(level=0):
                    diffs.append({"task": task, "comparison": f"{c} − mlp", "seeds": len(v),
                                  "ΔWGA (pp)": f"{v.mean() * 100:+.1f} ± {v.std() * 100 if len(v) > 1 else 0:.1f}"})
        if diffs:
            lines += [f"\nPaired worst-group difference vs MLP ({rule} selection):\n",
                      pd.DataFrame(diffs).to_markdown(index=False), "\n"]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))

    # figure: WGA per task and connector (wga selection), mean ± std over seeds
    d = df[df.rule == "wga"]
    tasks = sorted(d.task.unique())
    conns = [c for c in ("mlp", "qformer", "bottleneck") if c in set(d.connector)]
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(tasks)), 4.2), facecolor="#fcfcfb")
    width = 0.8 / len(conns)
    for i, c in enumerate(conns):
        stats = [d[(d.task == t) & (d.connector == c)].worst_group_acc for t in tasks]
        means = [s.mean() * 100 if len(s) else np.nan for s in stats]
        sds = [s.std() * 100 if len(s) > 1 else 0 for s in stats]
        x = np.arange(len(tasks)) + (i - (len(conns) - 1) / 2) * width
        ax.bar(x, means, width * 0.92, yerr=sds, capsize=3, color=SERIES[c], label=LABELS[c],
               error_kw={"ecolor": "#52514e", "elinewidth": 1})
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Test worst-group accuracy (%)", color="#52514e")
    ax.set_ylim(0, 105)
    ax.set_title("Worst-group accuracy by shortcut type and connector", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#e4e3df")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "wga_by_task.png"), dpi=200, facecolor="#fcfcfb")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
