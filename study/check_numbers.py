"""Recompute the paper's headline and control numbers from the saved per-run results, and check that each value
printed here appears in the manuscript.

Every comparison is paired by the true seed (folder s<k>). Reference arms (zero-start and fixed gate, Qwen3.5-4B) come
from results_study/h1/all_runs.csv, restricted to the seeds the control ran.

    python study/check_numbers.py            # table of recomputed values + manuscript check
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "study"))
from rounds_4_8 import runs  # noqa: E402

TASKS = ["waterbirds", "watermark_p0.95"]


def ref():
    h = pd.read_csv(os.path.join(ROOT, "results_study/h1/all_runs.csv"))
    h = h[h.rule == "wga"].copy()
    h["wga"] = 100 * h.worst_group_acc
    assert h.groupby(["task", "connector"]).seed.nunique().eq(15).all(), "h1 seeds must be the true seeds 0-14"
    return h[["task", "connector", "seed", "wga"]]


def csv(name):
    return pd.read_csv(os.path.join(ROOT, "results_study", name))


def control_sources():
    k = lambda job: runs(os.path.join(ROOT, "kaggle_out"), job)
    wo = pd.concat([csv("r11_wo_runs.csv"), csv("r13_wo34_runs.csv")])
    return [
        ("capped injection", csv("r9_capped_runs.csv"), "bottleneck_capped"),
        ("start 0.01 (W_o = 0)", k("r9-eps01"), "bottleneck_eps01"),
        ("start 0.1 (W_o = 0)", k("r9-eps1"), "bottleneck_eps1"),
        ("start 0.01, default W_o", wo, "bottleneck_eps01d"),
        ("start 0.1, default W_o", wo, "bottleneck_eps1d"),
        ("start 0.5 (half open)", k("r7-halfgate"), "bottleneck_half"),
        ("gate out of clipping", k("r5-controls").query("tag == 'qwen35-4b_clipx'"), "bottleneck"),
        ("no weight decay on gate", k("r5-controls").query("tag == 'qwen35-4b_nowd'"), "bottleneck"),
    ]


def pair(x, y):
    s = sorted(set(x.index) & set(y.index))
    d = y[s] - x[s]
    p = stats.ttest_rel(y[s], x[s]).pvalue if len(s) > 1 else np.nan
    return s, d.mean(), p


def controls():
    r = ref()
    rows = []
    for name, d, conn in control_sources():
        for t in TASKS:
            v = d[(d.task == t) & (d.connector == conn)].groupby("seed").wga.mean()
            if v.empty:
                continue
            z = r[(r.task == t) & (r.connector == "bottleneck")].set_index("seed").wga
            f = r[(r.task == t) & (r.connector == "bottleneck_fixed")].set_index("seed").wga
            s, dz_, pz = pair(z, v)
            _, df_, pf = pair(f, v)
            g = d[(d.task == t) & (d.connector == conn)].gate.abs() if "gate" in d else pd.Series(dtype=float)
            rows.append(dict(control=name, task=t, seeds=len(s), variant=v[s].mean(), zero=z[s].mean(), fixed=f[s].mean(),
                             vs_zero=dz_, p_zero=pz, vs_fixed=df_, p_fixed=pf, collapsed=int((v[s] < 50).sum()),
                             gate_min=g.min() if len(g) else np.nan, gate_max=g.max() if len(g) else np.nan))
    return pd.DataFrame(rows)


def default_gap():
    r = ref()
    out = {}
    for t in TASKS:
        z = r[(r.task == t) & (r.connector == "bottleneck") & (r.seed < 5)].set_index("seed").wga
        f = r[(r.task == t) & (r.connector == "bottleneck_fixed") & (r.seed < 5)].set_index("seed").wga
        out[t] = (z.mean(), f.mean(), (f - z).mean(), stats.ttest_rel(f, z).pvalue)
    return out


def manuscript():
    files = glob.glob(os.path.join(ROOT, "paper/manuscript/*.md")) + \
        [os.path.join(ROOT, "paper", f) for f in ("INTRODUCTION.md", "RELATED_WORK.md", "COVER_LETTER.md")] + \
        [os.path.join(ROOT, "paper/springer/main.tex")]
    return {os.path.relpath(f, ROOT): open(f).read() for f in files}


def main():
    pd.set_option("display.width", 200)
    c = controls()
    print("Controls, paired by true seed (two-sided paired t):")
    print(c.round(3).to_string(index=False))
    print("\nDefault setting, seeds 0-4 (zero start, fixed, gap, p two-sided):")
    for t, v in default_gap().items():
        print(f"  {t}: {v[0]:.1f} / {v[1]:.1f}, gap {v[2]:+.1f}, p = {v[3]:.3f}")
    text = "\n".join(manuscript().values())
    print("\nVariant means not found in the manuscript (formatted to one decimal):")
    missing = [f"{r.control} / {r.task}: {r.variant:.1f}" for r in c.itertuples() if f"{r.variant:.1f}" not in text]
    print("  " + ("\n  ".join(missing) if missing else "none"))
    stale = [v for v in ("61.2 / 42.6", "73.1 / 67.2", "64.4 / 39.3", "60.9 / 44.9", "+24.6 on the same", "+11.9 of the")
             if v in text]
    print("Stale reference values still in the manuscript:", stale or "none")
    return 1 if stale or missing else 0


if __name__ == "__main__":
    sys.exit(main())
