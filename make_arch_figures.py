"""Architecture figures.

    python make_arch_figures.py

Writes results/figures/fig14_architecture.{pdf,png} and
fig15_paper_vs_ours.{pdf,png}. Same palette and attribution as make_figures.py, so
they sit alongside the measured figures without looking bolted on.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

NAVY = "#122a56"
GOLD = "#b88a28"
TEAL = "#1f6f78"
GREY = "#787e86"
LIGHT = "#e8e9eb"
INK = "#2b2f33"

# fill, edge, text — one per provenance class
UNCHANGED = ("#f2f3f5", "#9aa0a6", INK)
SUBSTITUTED = ("#faeeda", GOLD, "#633806")
CONTRIB = ("#e1f5ee", TEAL, "#0f6e56")

AUTHOR = ("MD Humayun · supervised by Dr. M.A. Zaveri · "
          "M.Tech CSE (Information Security and Privacy), SVNIT Surat")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
})

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, sub=None, style=UNCHANGED, fs=9.5, subfs=8):
    fill, edge, tc = style
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=0.9, facecolor=fill, edgecolor=edge, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color=tc, alpha=0.8, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3)


def arrow(ax, x1, y1, x2, y2, style="-|>", rad=0.0, colour="#6b7076"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
        linewidth=0.9, color=colour, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


def elbow(ax, pts, colour="#6b7076"):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=colour, lw=0.9, zorder=1, solid_capstyle="round")
    arrow(ax, *pts[-2], *pts[-1], colour=colour)


def credit(fig, y=0.005):
    fig.text(0.005, y, AUTHOR, fontsize=5.6, color="#9aa0a6", ha="left", va="bottom")


# ===================================================================== FIG 14
def fig14():
    fig, ax = plt.subplots(figsize=(8.0, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    H = 6.6
    box(ax, 8, 90, 34, H, "Query", "RefCOCO / RefCOCO+ testA", UNCHANGED)
    box(ax, 50, 90, 44, H, "API prompt", "C1 released → C4 conditional", CONTRIB)
    arrow(ax, 25, 90, 40, 84.2)
    arrow(ax, 72, 90, 60, 84.2)

    box(ax, 25, 77.6, 50, H, "Program generator  π",
        "Qwen2.5-Coder 1.5B / 7B / 32B  ·  replaces Codex", SUBSTITUTED)
    arrow(ax, 50, 77.6, 50, 71.8)

    box(ax, 33, 65.2, 34, H, "programs.jsonl", "stage boundary, two environments", CONTRIB)
    arrow(ax, 50, 65.2, 50, 59.4)

    box(ax, 25, 52.8, 50, H, "Execution engine  φ", "restricted namespace, timeout", UNCHANGED)
    arrow(ax, 50, 52.8, 50, 47.0)

    box(ax, 18, 40.4, 64, H, "ImagePatch API",
        "find · verify_property · compute_depth", CONTRIB, subfs=7.6)
    ax.text(50, 39.1, "+ depth_order · is_behind · is_in_front_of · "
            "is_between_3d · distance_3d",
            ha="center", va="center", fontsize=7.6, color="#0f6e56", zorder=3)
    arrow(ax, 50, 40.4, 50, 34.6)

    box(ax, 25, 28.0, 50, H, "ModuleBus", "single dispatch point", UNCHANGED)

    arrow(ax, 38, 28.0, 18.5, 22.2)
    arrow(ax, 50, 28.0, 50, 22.2)
    arrow(ax, 62, 28.0, 81.5, 22.2)

    box(ax, 4, 15.6, 29, H, "GLIP-L", "find, exists", UNCHANGED, subfs=7.4)
    box(ax, 35.5, 15.6, 29, H, "CLIP ViT-L", "replaces X-VLM", SUBSTITUTED, subfs=7.4)
    box(ax, 67, 15.6, 29, H, "MiDaS DPT", "sign corrected", CONTRIB, subfs=7.4)

    elbow(ax, [(75, 56.1), (97, 56.1), (97, 6.5), (75.6, 6.5)])
    box(ax, 25, 3.2, 50, H, "Predicted box", "scored by IoU against ground truth", UNCHANGED)

    for i, (lbl, st) in enumerate([("unchanged from the paper", UNCHANGED),
                                   ("forced substitution", SUBSTITUTED),
                                   ("this work", CONTRIB)]):
        x = 8 + i * 30
        ax.add_patch(FancyBboxPatch((x, -3.4), 2.4, 1.8,
                                    boxstyle="round,pad=0,rounding_size=0.4",
                                    facecolor=st[0], edgecolor=st[1], lw=0.9))
        ax.text(x + 3.4, -2.5, lbl, fontsize=8, color=INK, va="center")
    ax.set_ylim(-5.5, 98)

    credit(fig, 0.002)
    fig.savefig(OUT / "fig14_architecture.pdf")
    fig.savefig(OUT / "fig14_architecture.png", dpi=200)
    plt.close(fig)
    print("  fig14_architecture")


# ===================================================================== FIG 15
def fig15():
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    rows = [
        ("Prompt", "one task-agnostic API spec",
         "four variants, C1 to C4", CONTRIB),
        ("Generator  π", "Codex code-davinci-002",
         "Qwen2.5-Coder, three sizes", SUBSTITUTED),
        ("Pipeline", "single process, producer–consumer",
         "two stages split at programs.jsonl", CONTRIB),
        ("Depth API", "compute_depth() → one scalar",
         "depth_order, is_behind, distance_3d, …", CONTRIB),
        ("Depth backend", "MiDaS, output used as returned",
         "MiDaS, sign corrected and resampled", CONTRIB),
        ("verify_property", "X-VLM", "CLIP ViT-L", SUBSTITUTED),
        ("simple_query / llm_query", "BLIP-2-XXL / GPT-3",
         "neutral fallback, out of scope", SUBSTITUTED),
        ("GLIP build", "torch 1.13, CUDA 11.6",
         "torch 2.1.2, CUDA 12.1, sm_90", SUBSTITUTED),
        ("Detection threshold", "0.5", "0.2, validated held out", CONTRIB),
        ("Scoring", "mean IoU; missing → mean box",
         "acc@0.5 and mean IoU; missing → 0", CONTRIB),
        ("Validation", "seeds not stated",
         "12 deviations, 3 seeds, 2 controls", CONTRIB),
    ]

    ax.text(27, 96, "ViperGPT  (Surís et al., ICCV 2023)", ha="center",
            fontsize=11, color=NAVY)
    ax.text(76, 96, "This work", ha="center", fontsize=11, color=TEAL)
    ax.plot([2, 98], [93.6, 93.6], color=LIGHT, lw=1.2)

    top, h, gap = 86.0, 6.4, 1.5
    for i, (comp, left, right, st) in enumerate(rows):
        y = top - i * (h + gap)
        ax.text(2, y + h / 2, comp, fontsize=8.6, color=INK, va="center")
        box(ax, 21, y, 30, h, "", None, UNCHANGED)
        ax.text(36, y + h / 2, left, ha="center", va="center",
                fontsize=8, color="#5c6167")
        box(ax, 60, y, 32, h, "", None, st)
        ax.text(76, y + h / 2, right, ha="center", va="center",
                fontsize=8, color=st[2])
        ax.annotate("", xy=(59, y + h / 2), xytext=(52, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#b9bcc0", lw=0.8,
                                    mutation_scale=8))

    ymin = top - (len(rows) - 1) * (h + gap) - 6
    ax.text(2, ymin, "Unchanged: the method itself — query → program → execute — "
                     "the executor semantics, GLIP as detector, and the task.",
            fontsize=8, color="#5c6167")
    ax.set_ylim(ymin - 4, 100)

    credit(fig, 0.004)
    fig.savefig(OUT / "fig15_paper_vs_ours.pdf")
    fig.savefig(OUT / "fig15_paper_vs_ours.png", dpi=200)
    plt.close(fig)
    print("  fig15_paper_vs_ours")


if __name__ == "__main__":
    print("writing to results/figures/")
    fig14()
    fig15()
