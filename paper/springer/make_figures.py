"""Journal artwork for the manuscript (IJCV rules: no titles inside figures, sans-serif 8 pt lettering, 84 or 174 mm
wide, patterns in addition to colour, vector PDF with embedded fonts).

    python paper/springer/make_figures.py
Bar-chart figures come from study/final_figures.py --journal; this script adds the paired-gain plot and the
per-seed plot from the committed results.
"""
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
MM = 1 / 25.4


def hypotheses(path, llm):
    rows = []
    for line in open(path):
        if line.startswith("| waterbirds") or line.startswith("| watermark"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            m = re.match(r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", c[4])
            rows.append((llm, "Waterbirds" if c[0] == "waterbirds" else "Watermark", int(c[1]),
                         float(m[1]), float(m[2]), float(m[3])))
    return rows


def scale9b():
    """Exploratory Qwen3.5-9B rows: paired gain over seeds both arms finished, t-based 95% CI. Waterbirds seeds 3-4
    come from round 13 (both arms in one session); the earlier unpaired round-10 learnable run of seed 3 is dropped."""
    from scipy import stats
    d = pd.read_csv(os.path.join(ROOT, "results_study/r10_scale9b_runs.csv"))
    d = d[~((d.task == "waterbirds") & (d.seed >= 3))]
    d = pd.concat([d, pd.read_csv(os.path.join(ROOT, "results_study/r13_9b_wb34_runs.csv"))])
    rows = []
    for task, name in [("waterbirds", "Waterbirds"), ("watermark_p0.95", "Watermark")]:
        t = d[d.task == task]
        a = t[t.connector == "bottleneck"].set_index("seed").wga
        b = t[t.connector == "bottleneck_fixed"].set_index("seed").wga
        s = sorted(set(a.index) & set(b.index))
        diff = (b[s] - a[s]).values
        h = stats.t.ppf(0.975, len(s) - 1) * diff.std(ddof=1) / np.sqrt(len(s))
        rows.append(("Qwen3.5-9B", name, len(s), diff.mean(), diff.mean() - h, diff.mean() + h))
    return rows


def forest():
    rows = hypotheses(os.path.join(ROOT, "results_study/hypotheses.md"), "Qwen3.5-4B") + \
        hypotheses(os.path.join(ROOT, "results_study/phi/hypotheses.md"), "Phi-4-mini") + scale9b()
    fig, ax = plt.subplots(figsize=(84 * MM, 58 * MM))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#444", lw=0.6)
    for yi, (llm, task, n, g, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color="#222", lw=1.2)
        ax.plot(g, yi, marker="^" if "9B" in llm else "s" if llm.startswith("Qwen") else "o",
                color="#222", mfc="white" if "9B" in llm else "#222", ms=4)
        ax.text(g, yi + 0.18, f"+{g:.1f}", ha="center", va="bottom", fontsize=7)  # above the marker, inside the axes
    ax.set_yticks(y, [f"{llm}\n{task}, n = {n}" for llm, task, n, *_ in rows], fontsize=7)
    ax.set_xlabel("Fixed minus learnable gate,\nworst-group accuracy (points)")
    ax.set_xlim(min(-2, min(r[4] for r in rows) - 2), max(r[5] for r in rows) + 3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forest_gain.pdf"))
    plt.close(fig)


def seed_pairs():
    d = pd.read_csv(os.path.join(ROOT, "results_study/h1/all_runs.csv"))
    d = d[d.rule == "wga"]
    d["wga"] = 100 * d.worst_group_acc
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 55 * MM), sharey=True)
    for ax, (task, name, letter) in zip(axes, [("waterbirds", "Waterbirds", "a"), ("watermark_p0.95", "Watermark", "b")]):
        t = d[d.task == task]
        lo = t[t.connector == "bottleneck"].reset_index(drop=True)
        fx = t[t.connector == "bottleneck_fixed"].reset_index(drop=True)
        for (_, a), (_, b) in zip(lo.iterrows(), fx.iterrows()):  # rows are paired runs in file order
            up = b.wga >= a.wga
            ax.plot([0, 1], [a.wga, b.wga], color="#222" if up else "#888", lw=0.8, ls="-" if up else "--",
                    marker="o", ms=3)
        ax.axhline(50, color="#aaa", lw=0.6, ls=":")
        ax.set_xticks([0, 1], ["Learnable gate", "Fixed gate"])
        ax.set_xlim(-0.3, 1.3)
        ax.text(0.01, 0.98, f"({letter}) {name}", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    axes[0].set_ylabel("Test worst-group accuracy (%)")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "h1_seeds_qwen.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "study/final_figures.py"), "--journal", "--results",
                    os.path.join(ROOT, "results_study"), "--out", OUT], check=True)
    forest()
    seed_pairs()
    print("journal figures written to", OUT)
