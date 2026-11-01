"""
Summary figures over every experiment of the connector study, from the all_runs.csv files that
study/aggregate.py writes into results_study/<experiment>/. Each bar is the mean test worst-group accuracy
over seeds (checkpoint chosen by validation WGA), each dot one seed.

    python study/final_figures.py --results results_study --out docs/figures/final
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"mlp": "#2a78d6", "qformer": "#eb6834", "bottleneck": "#1baf7a", "bottleneck_fixed": "#7a5bd6"}
LABELS = {"mlp": "MLP", "qformer": "Q-Former", "bottleneck": "Cross-attn\nlearned gate",
          "bottleneck_fixed": "Cross-attn\nfixed gate"}
TASKS = {"waterbirds": "Waterbirds (background)", "watermark_p0.95": "Watermark (synthetic)", "celeba": "CelebA (gender)"}


def load(results, exp):
    path = os.path.join(results, exp, "all_runs.csv") if exp else os.path.join(results, "all_runs.csv")
    df = pd.read_csv(path)
    df = df[df["rule"] == "wga"].copy()
    df["wga"] = df["worst_group_acc"] * 100
    return df


def bars(ax, groups, title):
    """groups: list of (label, color, per-seed values)."""
    for i, (lab, col, vals) in enumerate(groups):
        vals = np.asarray(vals, dtype=float)
        ax.bar(i, vals.mean(), color=col, alpha=0.35, width=0.7, edgecolor=col)
        ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, color=col, capsize=4, lw=1.4)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, color=col, s=14, zorder=3)
        ax.text(i, 2, f"{vals.mean():.1f}\nn={len(vals)}", ha="center", va="bottom", fontsize=7.5, color="#222")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=170)
    plt.close(fig)


def vals(df, task, conn):
    return df[(df.task == task) & (df.connector == conn)]["wga"].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_study")
    parser.add_argument("--out", default="docs/figures/final")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main_df, h1 = load(args.results, None), load(args.results, "h1")
    conns = ["mlp", "qformer", "bottleneck", "bottleneck_fixed"]

    # 1. Main result: four connectors, gate arms with every H1 seed
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, ["waterbirds", "watermark_p0.95"]):
        groups = [(LABELS[c], COLORS[c], vals(h1 if c.startswith("bottleneck") else main_df, task, c)) for c in conns]
        bars(ax, groups, TASKS[task])
    axes[0].set_ylabel("Test worst-group accuracy (%)")
    fig.suptitle("Qwen3.5-4B, SigLIP-base: only the learned gate fails", fontsize=11)
    save(fig, args.out, "main_connectors")

    # 2. Second LLM
    phi = load(args.results, "phi")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, ["waterbirds", "watermark_p0.95"]):
        bars(ax, [(LABELS[c], COLORS[c], vals(phi, task, c)) for c in conns], TASKS[task])
    axes[0].set_ylabel("Test worst-group accuracy (%)")
    fig.suptitle("Phi-4-mini (5 seeds): same ordering", fontsize=11)
    save(fig, args.out, "phi_connectors")

    # 3. Vision encoder: SigLIP-base vs SigLIP-large on Waterbirds (seeds 0-1 for large)
    large = load(args.results, "siglip_large")
    fig, ax = plt.subplots(figsize=(10, 4))
    groups = []
    for c in conns:
        base = vals(h1 if c.startswith("bottleneck") else main_df, "waterbirds", c)
        groups += [(LABELS[c] + "\nbase", COLORS[c], base), (LABELS[c] + "\nlarge", COLORS[c], vals(large, "waterbirds", c))]
    bars(ax, groups, "Waterbirds: SigLIP-base vs SigLIP-large features")
    ax.set_ylabel("Test worst-group accuracy (%)")
    save(fig, args.out, "encoder_scale")

    # 4. Fixes: connector-only DFR and group-balanced sampling, plus linear DFR on the same features
    s3, bal = load(args.results, "stage3"), load(args.results, "balanced")
    lin = pd.read_csv(os.path.join(args.results, "linear_baselines.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, ["waterbirds", "watermark_p0.95"]):
        groups = [("MLP", COLORS["mlp"], vals(s3, task, "mlp")),
                  ("MLP\n+ conn-DFR", COLORS["mlp"], vals(s3, task, "mlp_dfr")),
                  ("Fixed gate", COLORS["bottleneck_fixed"], vals(s3, task, "bottleneck_fixed")),
                  ("Fixed gate\n+ conn-DFR", COLORS["bottleneck_fixed"], vals(s3, task, "bottleneck_fixed_dfr"))]
        if task == "watermark_p0.95":
            groups += [("MLP\nbalanced*", "#555", vals(bal, task, "mlp")),
                       ("Fixed gate\nbalanced*", "#555", vals(bal, task, "bottleneck_fixed"))]
        l = lin[(lin.task == task) & (lin.method == "dfr")]
        groups.append(("Linear DFR\n(no LLM)", "#999", l["worst_group_acc"].to_numpy() * 100))
        bars(ax, groups, TASKS[task])
    axes[0].set_ylabel("Test worst-group accuracy (%)")
    fig.suptitle("Fixes. conn-DFR: connector retrained on a balanced half of val. *balanced sampling uses group labels",
                 fontsize=10)
    save(fig, args.out, "fixes")

    # 5. Ablations on Waterbirds with the MLP connector
    abl = [("Default\n49 tokens, 4 ep", vals(main_df, "waterbirds", "mlp")),
           ("36 tokens", vals(load(args.results, "mlp36"), "waterbirds", "mlp")),
           ("+ LTD caption\nloss", vals(load(args.results, "ltd"), "waterbirds", "mlp")),
           ("8 epochs", vals(load(args.results, "long8"), "waterbirds", "mlp"))]
    long8 = load(args.results, "long8")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, [(l, COLORS["mlp"], v) for l, v in abl] +
         [("Fixed gate\n4 ep", COLORS["bottleneck_fixed"], vals(h1, "waterbirds", "bottleneck_fixed")),
          ("Fixed gate\n8 ep", COLORS["bottleneck_fixed"], vals(long8, "waterbirds", "bottleneck_fixed"))],
         "Waterbirds ablations (token count, caption loss, training length)")
    ax.set_ylabel("Test worst-group accuracy (%)")
    save(fig, args.out, "ablations")

    # 6. CelebA
    cel = load(args.results, "celeba")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars(ax, [(LABELS[c], COLORS[c], vals(cel, "celeba", c)) for c in ["mlp", "bottleneck", "bottleneck_fixed"]],
         TASKS["celeba"] + ", Blond_Hair vs Male")
    ax.set_ylabel("Test worst-group accuracy (%)")
    save(fig, args.out, "celeba")
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
