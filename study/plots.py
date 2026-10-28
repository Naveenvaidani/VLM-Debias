"""
Publication figures for the connector study, from the all_runs.csv written by study/aggregate.py.

    python study/plots.py --csv runs_study/qwen35-4b/summary/all_runs.csv --out docs/figures
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONNECTORS = ["mlp", "qformer", "bottleneck", "bottleneck_fixed"]
LABELS = {"mlp": "MLP\nprojector", "qformer": "Q-Former", "bottleneck": "Cross-attn\n(learned gate)",
          "bottleneck_fixed": "Cross-attn\n(fixed gate)"}
COLORS = {"mlp": "#2a78d6", "qformer": "#eb6834", "bottleneck": "#1baf7a", "bottleneck_fixed": "#7a5bd6"}
TASK_TITLES = {"waterbirds": "Background\n(Waterbirds)", "cooccur": "Co-occurrence\n(COCO)",
               "watermark_p0.95": "Watermark\n(synthetic)", "typographic_p0.95": "Text in image\n(synthetic)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def wrap(text, fig):
    """Wrap a figure title to the figure width (~12 characters per inch at 12.5 pt)."""
    return "\n".join(textwrap.wrap(text, width=int(fig.get_figwidth() * 11)))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ordered(values, order):
    return [v for v in order if v in set(values)] + sorted(set(values) - set(order))


def bars_per_task(df, metric, ylabel, title, path, ylim=(0, 105)):
    """One panel per task; bars = mean over seeds, dots = individual seeds, label = mean."""
    tasks = ordered(df.task.unique(), list(TASK_TITLES))
    conns = ordered(df.connector.unique(), CONNECTORS)
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(3.4, 1.15 * len(conns)) * len(tasks), 4.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, task in zip(axes[0], tasks):
        for i, c in enumerate(conns):
            vals = df[(df.task == task) & (df.connector == c)][metric].to_numpy() * 100
            if not len(vals):
                continue
            mean, sd = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.bar(i, mean, 0.62, color=COLORS[c], alpha=0.9, zorder=2)
            ax.errorbar(i, mean, yerr=sd, color=INK2, capsize=4, lw=1, zorder=3)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=INK, alpha=0.55, zorder=4,
                       linewidths=0)
            ax.text(i, min(mean + sd + 2.5, ylim[1] - 4), f"{mean:.1f}", ha="center", va="bottom", fontsize=9,
                    color=INK)
        ax.set_xticks(range(len(conns)))
        ax.set_xticklabels([LABELS[c] for c in conns], fontsize=8.5)
        ax.set_title(TASK_TITLES.get(task, task), pad=10)
        ax.set_ylim(*ylim)
        style(ax)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(wrap(title, fig), x=0.01, ha="left", fontsize=12.5, color=INK, y=0.99)
    n_seeds = df.groupby(["task", "connector"]).seed.nunique().max()
    fig.text(0.01, 0.005, f"Bars: mean over {n_seeds} seed(s); error bars: ±1 s.d.; dots: individual seeds.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=2.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def id_vs_ood(df, path):
    """Dumbbell: in-distribution accuracy (circle) vs OOD accuracy (diamond); the gap is the shortcut cost."""
    tasks = ordered(df.task.unique(), list(TASK_TITLES))
    conns = ordered(df.connector.unique(), CONNECTORS)
    g = df.groupby(["task", "connector"])[["acc_id", "acc_ood"]].mean().reset_index()
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(3.4, 1.15 * len(conns)) * len(tasks), 4.2), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for i, c in enumerate(conns):
            row = g[(g.task == task) & (g.connector == c)]
            if row.empty:
                continue
            a_id, a_ood = row.acc_id.iloc[0] * 100, row.acc_ood.iloc[0] * 100
            ax.plot([i, i], [a_ood, a_id], color=COLORS[c], lw=3, alpha=0.5, zorder=2)
            ax.scatter([i], [a_id], s=60, color=COLORS[c], marker="o", zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([i], [a_ood], s=70, color=COLORS[c], marker="D", zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(i + 0.14, (a_id + a_ood) / 2, f"−{a_id - a_ood:.1f}", fontsize=8.5, color=INK2, va="center")
        ax.set_xticks(range(len(conns)))
        ax.set_xticklabels([LABELS[c] for c in conns], fontsize=8.5)
        ax.set_xlim(-0.6, len(conns) - 0.3)
        ax.set_title(TASK_TITLES.get(task, task), pad=10)
        ax.set_ylim(0, 105)
        style(ax)
    axes[0][0].set_ylabel("Test accuracy (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK2, label="In-distribution (majority groups)"),
               plt.Line2D([], [], marker="D", ls="", color=INK2, label="Out-of-distribution (minority groups)")]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(wrap("Shortcut cost: accuracy drop from ID to OOD groups (label = gap in points)", fig),
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=2.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def paired_delta(df, path):
    """Per-seed difference in worst-group accuracy vs the MLP baseline (same task, same seed)."""
    tasks = ordered(df.task.unique(), list(TASK_TITLES))
    others = [c for c in ("qformer", "bottleneck", "bottleneck_fixed") if c in set(df.connector)]
    piv = df.pivot_table(index=["task", "seed"], columns="connector", values="worst_group_acc")
    if "mlp" not in piv or not others:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(tasks) + 3.5, 4.6))
    width = 0.32
    lo, hi = 0.0, 0.0
    for j, c in enumerate(others):
        for i, task in enumerate(tasks):
            if task not in piv.index.get_level_values(0):
                continue
            d = ((piv.loc[task][c] - piv.loc[task]["mlp"]).dropna() * 100).to_numpy()
            if not len(d):
                continue
            x = i + (j - (len(others) - 1) / 2) * width
            ax.bar(x, d.mean(), width * 0.85, color=COLORS[c], alpha=0.9, zorder=2,
                   label=LABELS[c].replace("\n", " ") if i == 0 else None)
            ax.scatter(np.full(len(d), x), d, s=14, color=INK, alpha=0.55, zorder=3, linewidths=0)
            lo, hi = min(lo, d.min(), d.mean()), max(hi, d.max(), d.mean())
            ax.text(x + width * 0.47, d.mean(), f" {d.mean():+.1f}", ha="left", va="center", fontsize=8.5,
                    color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.axhline(0, color=INK2, lw=1)
    span = max(hi - lo, 4.0)
    ax.set_ylim(lo - 0.18 * span, hi + 0.22 * span)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_TITLES.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylabel("Δ worst-group accuracy vs MLP (pp)")
    fig.suptitle(wrap("Does the connector change shortcut reliance? Paired difference vs MLP, same task and seed "
                      "(positive = higher worst-group accuracy than MLP)", fig), x=0.01, ha="left", fontsize=12.5,
                 color=INK, y=0.99)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rule", default="wga", choices=["wga", "avg"], help="checkpoint-selection rule to plot")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df.rule == args.rule]
    sel = "val worst-group" if args.rule == "wga" else "val average (no group labels)"
    bars_per_task(df, "worst_group_acc", "Test worst-group accuracy (%)",
                  f"Worst-group accuracy by shortcut type and connector (selection: {sel})",
                  os.path.join(args.out, f"wga_{args.rule}.png"))
    bars_per_task(df, "avg_acc", "Test average accuracy (%)",
                  f"Average accuracy by shortcut type and connector (selection: {sel})",
                  os.path.join(args.out, f"avg_{args.rule}.png"))
    bars_per_task(df, "heldout_wga", "Worst-group accuracy, held-out question (%)",
                  "Generalisation to an unseen question phrasing", os.path.join(args.out, f"heldout_{args.rule}.png"))
    id_vs_ood(df, os.path.join(args.out, f"id_vs_ood_{args.rule}.png"))
    paired_delta(df, os.path.join(args.out, f"delta_vs_mlp_{args.rule}.png"))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
