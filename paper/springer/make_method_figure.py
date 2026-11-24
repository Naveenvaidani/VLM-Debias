"""Method figure (figures/method.pdf): data flow through the gated cross-attention connector and the gate variants.

Journal style: 174 mm wide, sans-serif 8 pt, no title inside the figure, vector PDF with embedded fonts.
Usage: python paper/springer/make_method_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "method.pdf")

fig = plt.figure(figsize=(6.85, 3.1))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 45)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ls="-", lw=0.8, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="black",
                                lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(x0, y0, x1, y1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, lw=0.8, color="black",
                                 shrinkA=0, shrinkB=0))


# inputs
box(1, 35, 11, 5, "Image")
box(1, 20, 11, 5, "Question")
box(15, 35, 15, 5, "SigLIP encoder\n(frozen)", fc="#e6e6e6")
box(15, 20, 15, 5, "LLM token\nembeddings", fc="#e6e6e6")
arrow(12.5, 37.5, 14.5, 37.5)
arrow(12.5, 22.5, 14.5, 22.5)

# connector region
ax.add_patch(FancyBboxPatch((33, 16.5), 38, 27, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f5f5f5",
                            ec="black", lw=0.9, ls="--"))
ax.text(52, 42.2, "Connector (the only variable)", ha="center", va="center", fontsize=8, fontweight="bold")
box(35, 34.5, 13, 5, "patches $P$\nLN, keys/values")
box(35, 22, 13, 5, "text $e_t$\nLN, $W_q$")
box(51, 28, 9, 6, "cross-\nattention")
box(62.5, 28, 6.5, 6, "$W_o$")
arrow(30.5, 37.5, 34.5, 37.5)
arrow(30.5, 22.5, 34.5, 24.5)
arrow(48.5, 37, 51, 32.5)
arrow(48.5, 24.5, 51, 29.5)
arrow(60.3, 31, 62.3, 31)
# gate and residual sum
ax.add_patch(plt.Circle((66, 22.5), 1.7, fc="white", ec="black", lw=0.8))
ax.text(66, 22.5, r"$\times g$", ha="center", va="center", fontsize=7)
arrow(65.75, 27.7, 65.9, 24.3)
ax.add_patch(plt.Circle((66, 18.6), 1.3, fc="white", ec="black", lw=0.8))
ax.text(66, 18.6, "+", ha="center", va="center", fontsize=9)
arrow(66, 20.8, 66, 19.95)
ax.plot([30.5, 32, 32, 64.6], [21.5, 21.5, 18.6, 18.6], color="black", lw=0.8)
ax.text(57.5, 19.5, r"$e_t' = e_t + g\,\Delta_t$", ha="center", va="bottom", fontsize=8)
ax.text(69.2, 25.3, r"$\Delta_t$", ha="left", va="center", fontsize=8)

# LLM and readout
box(75, 15.5, 13, 7, "LLM\n(4-bit, LoRA)", fc="#e6e6e6")
box(90.5, 15.5, 8.5, 7, "answer\ntoken")
arrow(67.4, 18.6, 74.5, 18.6)
arrow(88.6, 19, 90.2, 19)

# gate variants table
rows = [("Gate variant", r"gate $g$", r"start of $g$", r"$W_o$ at start", "trains $g$"),
        ("Learnable (default)", r"$\tanh\alpha$", "0", "random", "yes"),
        ("Fixed", "1", "1", "0", "no"),
        ("Small or open start", r"$\tanh\alpha$", "0.01, 0.1, 0.5, 0.995", "0", "yes"),
        ("Capped", r"1, with $\|\Delta_t\| \leq \|e_t\|$", "1", "0", "no")]
cols = [2, 24, 45, 66, 82]
y0 = 12.2
for i, r in enumerate(rows):
    y = y0 - i * 2.6
    for x, t in zip(cols, r):
        ax.text(x, y, t, ha="left", va="center", fontsize=7.5, fontweight="bold" if i == 0 else None)
ax.plot([2, 96], [y0 - 1.3, y0 - 1.3], color="black", lw=0.6)
ax.plot([2, 96], [y0 + 1.3, y0 + 1.3], color="black", lw=0.6)
ax.plot([2, 96], [y0 - 4 * 2.6 - 1.3, y0 - 4 * 2.6 - 1.3], color="black", lw=0.6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", ".png"), dpi=200)
print("wrote", OUT)
