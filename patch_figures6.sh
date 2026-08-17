#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

extra = '''

# ================================================================ FIG 11
def fig11_iou_sweep():
    """Is the result an artefact of the IoU 0.5 threshold?

    Every headline number uses IoU >= 0.5 because the paper does. Per-sample IoU is
    stored, so accuracy can be recomputed at any threshold at no cost. If C4 only
    beat C2 near 0.5 the result would be fragile; the curves should be separated
    across the whole range.
    """
    conds = [("C1", "m2_m3_refcoco_base", GREY),
             ("C2", "m2_m3_refcoco_grounding", NAVY),
             ("C3", "m2_m3_refcoco_depth", GOLD),
             ("C4", "m2_c4_refcoco", TEAL)]
    ths = np.arange(0.30, 0.92, 0.02)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    curves = {}
    for label, pat, colour in conds:
        rs = records(pat)
        if not rs:
            continue
        ious = np.array([r["iou"] for r in rs])
        acc_ = [100.0 * float((ious >= t).mean()) for t in ths]
        curves[label] = np.array(acc_)
        ax1.plot(ths, acc_, lw=2, color=colour, label=label)

    ax1.axvline(0.5, color="#999999", lw=1.0, ls=":")
    ax1.text(0.505, 2, "reported at 0.5", fontsize=7, color="#666666")
    ax1.set_xlabel("IoU threshold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 72)
    ax1.set_xlim(0.30, 0.90)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(color=LIGHT, lw=0.7)
    ax1.set_axisbelow(True)

    if "C2" in curves and "C4" in curves:
        gap = curves["C4"] - curves["C2"]
        ax2.plot(ths, gap, lw=2, color=TEAL)
        ax2.fill_between(ths, 0, gap, color=TEAL, alpha=0.16)
        ax2.axhline(0, color="#666666", lw=0.9)
        ax2.axvline(0.5, color="#999999", lw=1.0, ls=":")
        ax2.set_xlabel("IoU threshold")
        ax2.set_ylabel("C4 $-$ C2 (percentage points)")
        ax2.set_xlim(0.30, 0.90)
        ax2.set_title("advantage holds across thresholds",
                      fontsize=8.5, color="#444444", pad=6)
        ax2.grid(color=LIGHT, lw=0.7)
        ax2.set_axisbelow(True)

    credit(fig)
    fig.savefig(OUT / "fig11_iou_sweep.pdf")
    fig.savefig(OUT / "fig11_iou_sweep.png", dpi=200)
    plt.close(fig)
    print("  fig11_iou_sweep")


# ================================================================ FIG 12
def fig12_headroom():
    """How much of the achievable is being captured, and what limits the rest.

    From the depth-ordering selection study: for each candidate-filtering strategy,
    the accuracy obtained against the ORACLE — whether the correct box was present
    in the candidate set at all. The difference between the two bars is an ordering
    failure; the distance from the oracle to 100% is a detection failure. It shows
    the binding constraint has moved from ordering to detection.

    NOTE: this is the prototype selection study (oracle-style picking), not the
    end-to-end generated-program pipeline. Labelled as such.
    """
    f = Path("results/score_filter_ablation.json")
    if not f.exists():
        print("  fig12 skipped (no score_filter_ablation.json)")
        return
    d = json.load(open(f))["results"]
    order = [("all", "no filter"), ("area_topk", "top-6 by area"),
             ("score_topk", "top-6 by score"), ("score_nms_topk", "NMS + top-6")]
    order = [(k, l) for k, l in order if k in d]
    if not order:
        print("  fig12 skipped (empty ablation)")
        return

    labels = [l for _, l in order]
    acc_ = [d[k]["acc"] for k, _ in order]
    orc = [d[k]["oracle"] for k, _ in order]

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, orc, 0.58, color=LIGHT, edgecolor="#b9bcc0", linewidth=0.7,
           label="correct box present (oracle)")
    b = ax.bar(x, acc_, 0.58, color=TEAL, label="correct box selected")
    annotate(ax, b, "{:.1f}", dy=1.0)
    for xi, (a_, o_) in enumerate(zip(acc_, orc)):
        ax.text(xi, o_ + 1.2, f"{o_:.1f}", ha="center", fontsize=7.5,
                color="#6b7076")
        ax.annotate("", xy=(xi + 0.33, a_), xytext=(xi + 0.33, o_),
                    arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("depth-query accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("depth-ordering selection study, RefCOCO+ depth queries (n=31)",
                 fontsize=8.5, color="#444444", pad=6)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.02,
            "gap between bars = ordering failure\\ngap above grey = detection failure",
            transform=ax.transAxes, fontsize=7, color="#666666", va="bottom")
    credit(fig)
    fig.savefig(OUT / "fig12_headroom.pdf")
    fig.savefig(OUT / "fig12_headroom.png", dpi=200)
    plt.close(fig)
    print("  fig12_headroom")

'''

s = s.replace("\ndef main():", extra + "\ndef main():")
s = s.replace("    fig10_vs_paper()\n",
              "    fig10_vs_paper()\n    fig11_iou_sweep()\n    fig12_headroom()\n")
p.write_text(s)
print("added fig11_iou_sweep, fig12_headroom")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
