"""
Pre-declared tests of the gate hypotheses on all available seeds (learned vs fixed gate, paired by seed).

  H1  fixed gate > learned gate in test worst-group accuracy   one-sided paired t-test and Wilcoxon, alpha 0.025
  H2  learned gate varies more across seeds                     one-sided F-test on the seed variances

    python study/hypotheses.py --runs_dir runs_study/qwen35-4b --out results_study
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def seeds(runs_dir, task, conn):
    return {f.split(os.sep)[-2]: json.load(open(f))["test"]["wga"]["mean_over_templates"]["worst_group_acc"] * 100
            for f in glob.glob(os.path.join(runs_dir, task, conn, "s*", "results.json"))}


def figure(runs_dir, path):
    """Per-seed worst-group accuracy, learned vs fixed gate, one line per seed (paired)."""
    tasks = [t for t in sorted(os.listdir(runs_dir)) if seeds(runs_dir, t, "bottleneck")]
    titles = {"waterbirds": "Background (Waterbirds)", "watermark_p0.95": "Watermark (synthetic)"}
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4.2), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        lrn, fix = seeds(runs_dir, task, "bottleneck"), seeds(runs_dir, task, "bottleneck_fixed")
        common = sorted(set(lrn) & set(fix), key=lambda s: int(s[1:]))
        for s in common:
            ax.plot([0, 1], [lrn[s], fix[s]], color="#8a8986", lw=0.8, alpha=0.6, zorder=1)
        ax.scatter([0] * len(common), [lrn[s] for s in common], color="#1baf7a", s=22, zorder=2)
        ax.scatter([1] * len(common), [fix[s] for s in common], color="#7a5bd6", s=22, zorder=2)
        for x, vals in ((0, [lrn[s] for s in common]), (1, [fix[s] for s in common])):
            ax.plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color="#0b0b0b", lw=2.2, zorder=3)
        ax.axhline(50, color="#c8c7c2", lw=0.8, ls="--", zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Learned gate", "Fixed gate"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{titles.get(task, task)}\n{len(common)} seeds", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("Test worst-group accuracy (%)")
    fig.suptitle("Fixing the injection gate open: per-seed results (lines pair seeds; bars = mean)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lines = ["# Pre-declared gate hypotheses (test worst-group accuracy, %)\n",
             "| Task | Seeds | Learned gate | Fixed gate | Difference (95% CI) | H1 paired t | H1 Wilcoxon | "
             "H2 variance F | Collapsed seeds (< 50) learned / fixed |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for task in sorted(os.listdir(args.runs_dir)):
        lrn, fix = seeds(args.runs_dir, task, "bottleneck"), seeds(args.runs_dir, task, "bottleneck_fixed")
        common = sorted(set(lrn) & set(fix), key=lambda s: int(s[1:]))
        if len(common) < 3:
            continue
        l, f = np.array([lrn[s] for s in common]), np.array([fix[s] for s in common])
        d = f - l
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
        p_t = stats.ttest_rel(f, l, alternative="greater").pvalue
        p_w = stats.wilcoxon(f, l, alternative="greater").pvalue
        F = l.var(ddof=1) / f.var(ddof=1)
        p_f = 1 - stats.f.cdf(F, len(l) - 1, len(f) - 1)
        lines.append(f"| {task} | {len(common)} | {l.mean():.1f} ± {l.std(ddof=1):.1f} | {f.mean():.1f} ± {f.std(ddof=1):.1f} "
                     f"| {d.mean():+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] | {p_t:.1e} | {p_w:.1e} | {F:.1f} (p = {p_f:.1e}) "
                     f"| {int((l < 50).sum())} / {int((f < 50).sum())} |")
    os.makedirs(args.out, exist_ok=True)
    figure(args.runs_dir, os.path.join(args.out, "h1_gate_seeds.png"))
    with open(os.path.join(args.out, "hypotheses.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
