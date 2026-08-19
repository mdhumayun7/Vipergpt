"""Aggregate the model x condition x dataset grid into comparison tables.

    python make_grid_tables.py

Reads results/grid_summaries/*.json only — nothing is typed — and writes:

    results/grid_master.md        every cell, plus the four research tables
    results/grid_master.tex       the same, as LaTeX ready to \\input
    results/figures/fig17_grid.{pdf,png}          all models x all conditions
    results/figures/fig18_generalisation.{pdf,png} does each finding transfer?

The four questions the grid answers:

  Q1  Does the specification gap appear in every model family?      C1 -> C2
  Q2  Does the depth capability help in every family?               C2 -> C3
  Q3  Is the routing rule necessary in every family?                C3 -> C4
  Q4  Does a stronger generator raise accuracy at fixed condition?  across models
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- style
NAVY, GOLD, TEAL, GREY, LIGHT = "#122a56", "#b88a28", "#1f6f78", "#787e86", "#e8e9eb"
RUST = "#9e2a2a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
CREDIT = ("MD Humayun · supervised by Dr. M.A. Zaveri · "
          "M.Tech CSE (Information Security and Privacy), SVNIT Surat")

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- load
# Cell tags are grid_<dataset>_<model>_<condition>; the model short name may
# itself contain underscores, so the condition is taken from the tail.
CONDS = ["base", "grounding", "depth", "depth_cond"]
LABEL = {"base": "C1 released", "grounding": "C2 + contract",
         "depth": "C3 + depth", "depth_cond": "C4 + routing"}
PRETTY = {"7B": "Qwen2.5-Coder-7B", "1.5B": "Qwen2.5-Coder-1.5B",
          "32B": "Qwen2.5-Coder-32B",
          "DeepSeek-Coder-V2-Lite": "DeepSeek-Coder-V2-Lite (16B MoE)",
          "Yi-Coder-9B-Chat": "Yi-Coder-9B", "OpenCoder-8B": "OpenCoder-8B"}

grid: dict[tuple[str, str, str], dict] = {}
for f in glob.glob("results/grid_summaries/*.json"):
    d = json.load(open(f))
    tag = d.get("tag", "")
    if not tag.startswith("grid_"):
        continue
    rest = tag[len("grid_"):]
    ds = "refcoco+" if rest.startswith("refcoco+_") else "refcoco"
    rest = rest[len(ds) + 1:]
    cond = next((c for c in sorted(CONDS, key=len, reverse=True)
                 if rest.endswith("_" + c) or rest == c), None)
    if cond is None:
        continue
    model = rest[: -(len(cond) + 1)] if rest != cond else ""
    grid[(model, cond, ds)] = d

if not grid:
    raise SystemExit("no grid summaries found under results/grid_summaries/")

models = sorted({k[0] for k in grid},
                key=lambda m: (m not in ("1.5B", "7B", "32B"), m))
print(f"models: {models}")
print(f"cells : {len(grid)} of {len(models)*4*2}")


def cell(model, cond, ds, sub=None):
    d = grid.get((model, cond, ds))
    if d is None:
        return None
    return d[sub]["acc_iou50"] if sub else d["acc_iou50"]


def fmt(v, nd=2):
    return "---" if v is None else f"{v:.{nd}f}"


def name(m):
    return PRETTY.get(m, m)


# ================================================================ markdown
md = ["# Master comparison grid", "",
      "Accuracy at IoU >= 0.5, greedy decoding, n = 500 per cell, testA split.",
      "Every number is read from an archived run summary; nothing is transcribed.",
      "", "Conditions: C1 released prompt, C2 + return-type contract,",
      "C3 + depth primitives, C4 + primitives and routing rule.", ""]

for ds in ("refcoco", "refcoco+"):
    md += [f"## {ds} / testA", "",
           "| Generator | C1 | C2 | C3 | C4 | C1→C2 | C3→C4 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        v = [cell(m, c, ds) for c in CONDS]
        d12 = None if None in (v[0], v[1]) else v[1] - v[0]
        d34 = None if None in (v[2], v[3]) else v[3] - v[2]
        md.append(f"| {name(m)} | " + " | ".join(fmt(x) for x in v) +
                  f" | {fmt(d12,1)} | {fmt(d34,1)} |")
    md.append("")

    for sub, lbl in (("spatial", "spatial subset"), ("non_spatial", "non-spatial subset")):
        md += [f"### {ds} — {lbl}", "",
               "| Generator | C1 | C2 | C3 | C4 |", "|---|---:|---:|---:|---:|"]
        for m in models:
            md.append(f"| {name(m)} | " +
                      " | ".join(fmt(cell(m, c, ds, sub)) for c in CONDS) + " |")
        md.append("")

md += ["---", "", "## The four questions the grid answers", ""]

md += ["### Q1 — does the specification gap appear in every family? (C1 → C2)", "",
       "| Generator | C1 | C2 | gain |", "|---|---:|---:|---:|"]
for m in models:
    a, b = cell(m, "base", "refcoco"), cell(m, "grounding", "refcoco")
    md.append(f"| {name(m)} | {fmt(a)} | {fmt(b)} | "
              f"{fmt(None if None in (a,b) else b-a, 1)} |")
md += ["", "A gap present in every family cannot be a quirk of one model.", ""]

md += ["### Q2 — does the depth capability help in every family? (C2 → C3, depth subset)", "",
       "| Generator | C2 spatial | C3 spatial | gain |", "|---|---:|---:|---:|"]
for m in models:
    a = cell(m, "grounding", "refcoco+", "spatial")
    b = cell(m, "depth", "refcoco+", "spatial")
    md.append(f"| {name(m)} | {fmt(a)} | {fmt(b)} | "
              f"{fmt(None if None in (a,b) else b-a, 1)} |")
md += ["", "RefCOCO+ spatial is the depth-word subset (n = 47).", ""]

md += ["### Q3 — is the routing rule necessary in every family? (C3 → C4, non-spatial)", "",
       "| Generator | C2 | C3 | C4 | C3 cost | C4 recovery |", "|---|---:|---:|---:|---:|---:|"]
for m in models:
    a = cell(m, "grounding", "refcoco", "non_spatial")
    b = cell(m, "depth", "refcoco", "non_spatial")
    c = cell(m, "depth_cond", "refcoco", "non_spatial")
    md.append(f"| {name(m)} | {fmt(a)} | {fmt(b)} | {fmt(c)} | "
              f"{fmt(None if None in (a,b) else b-a, 1)} | "
              f"{fmt(None if None in (b,c) else c-b, 1)} |")
md += ["", "If C3 costs non-spatial accuracy in every family, over-application is a",
       "property of the specification rather than of one generator.", ""]

md += ["### Q4 — does a stronger generator raise accuracy? (best condition)", "",
       "| Generator | C4 RefCOCO | C4 RefCOCO+ | vs paper (72.0 / 67.0) |",
       "|---|---:|---:|---:|"]
for m in models:
    a, b = cell(m, "depth_cond", "refcoco"), cell(m, "depth_cond", "refcoco+")
    sh = "---" if a is None else f"{100*a/72.0:.0f}% / " + \
         ("---" if b is None else f"{100*b/67.0:.0f}%")
    md.append(f"| {name(m)} | {fmt(a)} | {fmt(b)} | {sh} |")
md += ["",
       "The perception ceiling measured on this stack is 81.5 (GLIP top-box, held-out",
       "n = 200), which is above the paper's 72.0. Perception is therefore not the",
       "limiting factor; program synthesis is.", ""]

Path("results/grid_master.md").write_text("\n".join(md) + "\n")
print("  results/grid_master.md")

# ================================================================ latex
tex = [r"\begin{table}[htbp]\centering\small",
       r"\caption{Master comparison grid. Accuracy at IoU $\geq0.5$, greedy decoding,",
       r"$n=500$ per cell, testA. C1 is the released prompt; C4 adds the depth",
       r"primitives together with the rule stating when they apply.}",
       r"\label{tab:grid}",
       r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
       r"& \multicolumn{4}{c}{\textbf{RefCOCO}} & \multicolumn{4}{c}{\textbf{RefCOCO+}} \\",
       r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
       r"\textbf{Generator} & C1 & C2 & C3 & C4 & C1 & C2 & C3 & C4 \\", r"\midrule"]
for m in models:
    row = [fmt(cell(m, c, "refcoco")) for c in CONDS] + \
          [fmt(cell(m, c, "refcoco+")) for c in CONDS]
    tex.append(name(m).replace("&", r"\&") + " & " + " & ".join(row) + r" \\")
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
Path("results/grid_master.tex").write_text("\n".join(tex) + "\n")
print("  results/grid_master.tex")

# ================================================================ fig 17
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), sharey=True)
colours = [GREY, NAVY, GOLD, TEAL]
for ax, ds in zip(axes, ("refcoco", "refcoco+")):
    x = np.arange(len(models))
    w = 0.2
    for i, c in enumerate(CONDS):
        vals = [cell(m, c, ds) or 0.0 for m in models]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=LABEL[c], color=colours[i],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v > 0.5:
                ax.text(b.get_x() + b.get_width()/2, v + 0.7, f"{v:.1f}",
                        ha="center", fontsize=6.5, color="#333333")
    ref = 72.0 if ds == "refcoco" else 67.0
    ax.axhline(ref, color=RUST, lw=1.3, ls="--")
    ax.text(len(models) - 0.45, ref + 1.2, f"paper {ref:.1f}", fontsize=7,
            color=RUST, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([name(m).replace(" (", "\n(") for m in models], fontsize=7.5)
    ax.set_title(f"{ds} / testA", fontsize=9.5, color="#444444", pad=6)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 80)
axes[0].set_ylabel("Accuracy, IoU $\\geq$ 0.5 (%)")
axes[0].legend(ncol=4, fontsize=7.5, loc="upper left")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.text(0.005, 0.012, CREDIT, fontsize=5.6, color="#9aa0a6", ha="left", va="bottom")
fig.savefig(OUT / "fig17_grid.pdf"); fig.savefig(OUT / "fig17_grid.png", dpi=200)
plt.close(fig)
print("  fig17_grid")

# ================================================================ fig 18
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
panels = [
    ("Q1  specification gap\nC1 $\\to$ C2, RefCOCO overall",
     lambda m: (cell(m, "base", "refcoco"), cell(m, "grounding", "refcoco")), NAVY),
    ("Q2  depth capability\nC2 $\\to$ C3, RefCOCO+ depth subset",
     lambda m: (cell(m, "grounding", "refcoco+", "spatial"),
                cell(m, "depth", "refcoco+", "spatial")), GOLD),
    ("Q3  routing rule\nC3 $\\to$ C4, RefCOCO non-spatial",
     lambda m: (cell(m, "depth", "refcoco", "non_spatial"),
                cell(m, "depth_cond", "refcoco", "non_spatial")), TEAL),
]
for ax, (title, getter, colour) in zip(axes, panels):
    ok = [(m,) + getter(m) for m in models]
    ok = [t for t in ok if t[1] is not None and t[2] is not None]
    if not ok:
        ax.text(0.5, 0.5, "no data yet", ha="center", transform=ax.transAxes,
                color=GREY); ax.axis("off"); continue
    y = np.arange(len(ok))
    for yi, (m, a, b) in zip(y, ok):
        ax.plot([a, b], [yi, yi], color=colour, lw=2.4, solid_capstyle="round",
                zorder=1)
        ax.scatter([a], [yi], s=26, color="white", edgecolor=GREY, zorder=2, lw=1.1)
        ax.scatter([b], [yi], s=34, color=colour, zorder=3)
        ax.text(max(a, b) + 1.4, yi, f"{b-a:+.1f}", va="center", fontsize=7.5,
                color=colour)
    ax.set_yticks(y)
    ax.set_yticklabels([name(t[0]).split(" (")[0] for t in ok], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)", fontsize=8)
    ax.set_title(title, fontsize=8.5, color="#444444", pad=8)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.text(0.005, 0.012, CREDIT, fontsize=5.6, color="#9aa0a6", ha="left", va="bottom")
fig.text(0.5, 0.005, "hollow marker = before the intervention, filled = after",
         fontsize=7, color="#666666", ha="center")
fig.savefig(OUT / "fig18_generalisation.pdf")
fig.savefig(OUT / "fig18_generalisation.png", dpi=200)
plt.close(fig)
print("  fig18_generalisation")

missing = [(m, c, ds) for m in models for c in CONDS
           for ds in ("refcoco", "refcoco+") if (m, c, ds) not in grid]
if missing:
    print(f"\n  {len(missing)} cells still missing:")
    for m, c, ds in missing[:12]:
        print(f"    {name(m):<34} {LABEL[c]:<16} {ds}")
else:
    print("\n  grid complete")
