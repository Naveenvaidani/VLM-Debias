"""Supplementary material: per-seed tables generated from the saved runs, so every number can be traced.

    python paper/springer/make_supplement.py      # writes supplement/supplement.tex and builds supplement.pdf
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "supplement")
sys.path.insert(0, os.path.join(ROOT, "study"))
from rounds_4_8 import runs  # noqa: E402
from check_numbers import control_sources, ref  # noqa: E402

TASK = {"waterbirds": "Waterbirds", "watermark_p0.95": "Watermark", "celeba": "CelebA"}
CONN = {"bottleneck": "zero start", "bottleneck_fixed": "fixed", "mlp": "MLP", "qformer": "Q-Former"}
KO = os.path.join(ROOT, "kaggle_out")


def f1(x):
    return "--" if pd.isna(x) else f"{x:.1f}"


def table(df, cols, caption, label, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    head = " & ".join(cols) + r" \\ \midrule"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False))
    return (f"\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n"
            f"\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}\n")


def paired_rows(d, a="bottleneck", b="bottleneck_fixed"):
    """One row per seed: zero, fixed, difference; then mean row."""
    za = d[d.connector == a].groupby("seed").wga.mean()
    zb = d[d.connector == b].groupby("seed").wga.mean()
    s = sorted(set(za.index) | set(zb.index))
    rows = [(k, f1(za.get(k)), f1(zb.get(k)), f1(zb.get(k, np.nan) - za.get(k, np.nan))) for k in s]
    p = sorted(set(za.index) & set(zb.index))
    rows.append((r"\textit{mean, paired}", f1(za[p].mean()), f1(zb[p].mean()), f1((zb[p] - za[p]).mean())))
    return pd.DataFrame(rows)


def h1_tables():
    out = []
    h = ref().rename(columns={"wga": "wga"})
    for llm, d in [("Qwen3.5-4B", h)]:
        for t in ("waterbirds", "watermark_p0.95"):
            out.append(table(paired_rows(d[d.task == t]), ["Seed", "Zero start", "Fixed", "Fixed $-$ zero"],
                             f"{llm}, {TASK[t]}: test worst-group accuracy (\\%) per seed (gate comparison, H1).",
                             f"tab:s-h1-{t[:5]}"))
    p = pd.read_csv(os.path.join(ROOT, "results_study/phi/all_runs.csv"))
    p = p[p.rule == "wga"].assign(wga=lambda x: 100 * x.worst_group_acc)
    for t in ("waterbirds", "watermark_p0.95"):
        out.append(table(paired_rows(p[p.task == t]), ["Seed", "Zero start", "Fixed", "Fixed $-$ zero"],
                         f"Phi-4-mini, {TASK[t]}: test worst-group accuracy (\\%) per seed.", f"tab:s-phi-{t[:5]}"))
    return out


def scale9b():
    d = pd.read_csv(os.path.join(ROOT, "results_study/r10_scale9b_runs.csv"))
    extra = pd.read_csv(os.path.join(ROOT, "results_study/r13_9b_wb34_runs.csv"))
    eps = pd.read_csv(os.path.join(ROOT, "results_study/r14_9b_eps1_wb_runs.csv"))
    out = []
    for t in ("waterbirds", "watermark_p0.95"):
        x = d[d.task == t]
        if t == "waterbirds":
            unpaired = x[(x.connector == "bottleneck") & (x.seed == 3)].wga
            x = pd.concat([x[x.seed < 3], extra[extra.task == t], eps[eps.task == t]])
        w = x.pivot_table(index="seed", columns="connector", values="wga", aggfunc="mean")
        cols = [c for c in ("bottleneck", "bottleneck_fixed", "bottleneck_eps1", "bottleneck_capped") if c in w]
        names = {"bottleneck": "Zero start", "bottleneck_fixed": "Fixed", "bottleneck_eps1": "Start 0.1",
                 "bottleneck_capped": "Capped"}
        df = w[cols].map(f1).reset_index()
        df.loc[len(df)] = [r"\textit{mean}"] + [f1(w[c].mean()) for c in cols]
        note = (" Seeds 3 and 4 were run in a later session (both arms together); an earlier, unpaired zero-start run "
                f"of seed 3 ({f1(unpaired.iloc[0])}) is not shown." if t == "waterbirds" else "")
        out.append(table(df, ["Seed"] + [names[c] for c in cols],
                         f"Qwen3.5-9B (exploratory), {TASK[t]}: test worst-group accuracy (\\%) per seed.{note}",
                         f"tab:s-9b-{t[:5]}"))
    return out


def controls():
    r = ref()
    rows = []
    for name, d, conn in control_sources():
        for t in ("waterbirds", "watermark_p0.95"):
            v = d[(d.task == t) & (d.connector == conn)].groupby("seed").wga.mean()
            for s, val in v.items():
                z = r[(r.task == t) & (r.connector == "bottleneck") & (r.seed == s)].wga
                fx = r[(r.task == t) & (r.connector == "bottleneck_fixed") & (r.seed == s)].wga
                rows.append((name.replace("W_o", "$W_o$"), TASK[t], s, f1(val), f1(z.mean() if len(z) else np.nan),
                             f1(fx.mean() if len(fx) else np.nan)))
    df = pd.DataFrame(rows)
    return table(df, ["Control", "Task", "Seed", "Variant", "Zero start", "Fixed"],
                 "Controls on Qwen3.5-4B compared with the reference arms of the same seed (test worst-group "
                 "accuracy, \\%). The reference arms come from the 15-seed gate comparison.", "tab:s-controls",
                 "llrrrr")


def own_pairs():
    jobs = [("r13-lr300", "Alignment 300 steps at $3\\times10^{-4}$"), ("r14-lr300-wm", "Alignment 300 steps at $3\\times10^{-4}$"),
            ("r12-long-wb", None), ("r14-lora-wm", None), ("r14-frac-wm", None)]
    label = {"qwen35-4b_lr3e4": "Alignment lr $3\\times10^{-4}$", "qwen35-4b_r4": "LoRA rank 4",
             "qwen35-4b_r64": "LoRA rank 64", "qwen35-4b_frac50": "Half training set",
             "qwen35-4b_frac25": "Quarter training set"}
    rows = []
    for job, _ in jobs:
        try:
            d = runs(KO, job)
        except Exception:
            continue
        for (tag, t), g in d.groupby(["tag", "task"]):
            if tag not in label:
                continue
            for r in paired_rows(g).itertuples(index=False):
                rows.append((label[tag], TASK[t]) + tuple(r))
    return table(pd.DataFrame(rows), ["Control", "Task", "Seed", "Zero start", "Fixed", "Fixed $-$ zero"],
                 "Controls whose two arms ran in the same job (Qwen3.5-4B, test worst-group accuracy, \\%).",
                 "tab:s-own", "llrrrr")


def long_align():
    d = pd.concat([pd.read_csv(os.path.join(ROOT, f"results_study/{f}.csv"))
                   for f in ("r12_long_align_runs", "r13_long6k_wb_runs")])
    rows = []
    for (tag, t), g in d.groupby(["tag", "task"]):
        n = tag.split("long")[1]
        for r in paired_rows(g).itertuples(index=False):
            rows.append((f"{int(n):,} steps", TASK[t]) + tuple(r) +
                        ((f1(g[(g.connector == 'bottleneck') & (g.seed == r[0])].gate.abs().max() * 1000),)
                         if isinstance(r[0], (int, np.integer)) else ("",)))
    return table(pd.DataFrame(rows), ["Alignment", "Task", "Seed", "Zero start", "Fixed", "Fixed $-$ zero",
                                      r"Gate $\times 10^3$"],
                 "Longer alignment at learning rate $3\\times10^{-4}$ (Qwen3.5-4B, test worst-group accuracy, \\%). "
                 "Gate: final $|\\tanh\\alpha|$ of the zero-start arm.", "tab:s-align", "llrrrrr")


def dfr_jtt():
    d = pd.concat([pd.read_csv(os.path.join(ROOT, f"results_study/r13_dfrjtt_{t}_runs.csv")) for t in ("wb", "wm")])
    cols = ["bottleneck", "bottleneck_dfr", "bottleneck_jtt", "bottleneck_fixed", "bottleneck_fixed_dfr",
            "bottleneck_fixed_jtt"]
    out = []
    for t, g in d.groupby("task"):
        w = g.pivot_table(index="seed", columns="connector", values="wga", aggfunc="mean")[cols]
        df = w.map(f1).reset_index()
        df.loc[len(df)] = [r"\textit{mean}"] + [f1(w[c].mean()) for c in cols]
        out.append(table(df, ["Seed", "Zero", "Zero + DFR", "Zero + JTT", "Fixed", "Fixed + DFR", "Fixed + JTT"],
                         f"Connector-level DFR and JTT on both gates, {TASK[t]} (Qwen3.5-4B, 3 epochs of "
                         "connector retraining, test worst-group accuracy, \\%).", f"tab:s-dfr-{t[:5]}"))
    return out


def celeba():
    o = pd.read_csv(os.path.join(ROOT, "results_study/celeba/all_runs.csv"))
    o = o[o.rule == "wga"]
    o = pd.DataFrame(dict(connector=o.connector, seed=o.seed, wga=100 * o.worst_group_acc))
    parts = [o, runs(KO, "r7-celeba")[["connector", "seed", "wga"]]]
    for job in ("r14-celeba-a", "r14-celeba-b", "r15-celeba-a", "r15-celeba-b"):
        if os.path.isdir(os.path.join(KO, job)):
            parts.append(runs(KO, job)[["connector", "seed", "wga"]])
    d = pd.concat(parts)
    return table(paired_rows(d), ["Seed", "Zero start", "Fixed", "Fixed $-$ zero"],
                 "CelebA (Qwen3.5-4B): test worst-group accuracy (\\%) per seed.", "tab:s-celeba")


def toy():
    d = pd.read_csv(os.path.join(ROOT, "results_study/toy_injection.csv"))
    d["wga"] = 100 * d.wga
    g = d.groupby(["opt", "steps", "gain"]).wga.agg(["mean", "std"]).reset_index()
    w = g.assign(v=g["mean"].map("{:.1f}".format) + " (" + g["std"].map("{:.1f}".format) + ")")
    w = w.pivot_table(index=["opt", "steps"], columns="gain", values="v", aggfunc="first").reset_index()
    w["opt"] = w.opt.map({"adam": "Adam", "sgd": "SGD"})
    gains = [c for c in w.columns if c not in ("opt", "steps")]
    return table(w, ["Optimiser", "Steps"] + [f"$g$={c:g}" for c in gains],
                 "Toy model, full grid: worst-group accuracy (\\%), mean (s.d.) over 20 seeds.", "tab:s-toy",
                 "lr" + "r" * len(gains))


def compute():
    c = pd.read_csv(os.path.join(ROOT, "results_study/compute_table.csv"))
    c = c.map(lambda v: f"{v:.2f}" if isinstance(v, float) else str(v).replace("_", r"\_").replace("%", r"\%"))
    return table(c, [str(x).replace("_", " ") for x in c.columns], "Compute cost per connector (one T4, batch 16).",
                 "tab:s-compute", "l" + "r" * (len(c.columns) - 1))


PRE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,longtable,amsmath,helvet}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\LTcapwidth}{\textwidth}
\title{Supplementary material: A Zero-Initialised Injection Gate Makes Shortcut Learning More Likely in
Vision-Language Models}
\date{}
\begin{document}
\maketitle
\noindent All tables in this document are generated from the saved per-run results by
\texttt{paper/springer/make\_supplement.py}; the comparisons in the paper are recomputed by
\texttt{study/check\_numbers.py}. Worst-group accuracy is the test accuracy of the worst label--attribute group,
with the checkpoint chosen by validation worst-group accuracy. "Zero start" is the learnable $\tanh$ gate started
at $\alpha_0 = 0$; "fixed" is the gate fixed at 1 with the output layer started at zero. Seeds are random
replicates: GPU kernels are not deterministic, so a seed does not replay a run exactly. Runs below 50\% count as
collapsed. Every finished run is shown.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    s = [PRE, r"\section{Gate comparison per seed}"] + h1_tables()
    s += [r"\section{Larger language model}"] + scale9b()
    s += [r"\section{Controls per seed}", controls(), own_pairs(), long_align()]
    s += [r"\section{Standard fixes on both gates}"] + dfr_jtt()
    s += [r"\section{CelebA per seed}", celeba()]
    s += [r"\section{Toy model}", r"Settings as in the paper: core feature $\pm 0.5$ (noise s.d.\ 0.1), shortcut "
          r"$\pm 2$ (s.d.\ 1) agreeing with the label in 95\% of training examples; Adam learning rate 0.01, SGD 0.5; "
          r"worst group on a balanced test set.", toy()]
    s += [r"\section{Compute}", compute(), r"\end{document}"]
    open(os.path.join(OUT, "supplement.tex"), "w").write("\n".join(s))
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplement.tex"], cwd=OUT,
                       stdout=subprocess.DEVNULL)
    print("written", os.path.join(OUT, "supplement.pdf"))


if __name__ == "__main__":
    main()
