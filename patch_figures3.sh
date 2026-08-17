#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

new_figs = '''

# ================================================================ FIG 7
def fig7_failure_modes():
    """Decompose every accuracy number into its failure causes.

    The headline figures say a condition scores 42%. This says WHY the other 58%
    fails, and the causes are qualitatively different: C1 fails almost entirely on
    return type, C4 mostly on localisation. A localisation miss is a perception
    limit; a wrong return type is a specification bug. Collapsing both into one
    accuracy number hides that distinction.
    """
    conds = [("C1", "m2_m3_refcoco_base"), ("C2", "m2_m3_refcoco_grounding"),
             ("C3", "m2_m3_refcoco_depth"), ("C4", "m2_c4_refcoco")]
    cats = ["correct (IoU $\\\\geq$ 0.5)", "box returned, IoU < 0.5",
            "wrong return type", "no execute_command", "runtime error", "timeout"]
    colours = [TEAL, "#8fbec2", GOLD, "#d9b45e", RUST, "#c98a8a"]

    rows, labels = [], []
    for label, pat in conds:
        rs = records(pat)
        if not rs:
            continue
        c = dict.fromkeys(cats, 0)
        for r in rs:
            err = r.get("error") or ""
            if r["kind"] == "patch":
                c[cats[0 if r["iou"] >= 0.5 else 1]] += 1
            elif "timeout" in err:
                c[cats[5]] += 1
            elif "did not define" in err:
                c[cats[3]] += 1
            elif r["kind"] in ("str", "bool", "list"):
                c[cats[2]] += 1
            else:
                c[cats[4]] += 1
        n = len(rs)
        rows.append([100.0 * c[k] / n for k in cats])
        labels.append(label)
    if not rows:
        print("  fig7 skipped (no records)")
        return

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for ci, (cat, colour) in enumerate(zip(cats, colours)):
        vals = np.array([r[ci] for r in rows])
        ax.barh(y, vals, 0.62, left=left, label=cat, color=colour,
                edgecolor="white", linewidth=0.6)
        for yi, (v, l_) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l_ + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if ci in (0, 2, 4) else "#333333")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of 500 samples (%)")
    ax.set_xlim(0, 100)
    ax.set_title("RefCOCO / testA — what happens to each sample",
                 fontsize=9, color="#444444", pad=6)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig, reserve=0.24)
    fig.savefig(OUT / "fig7_failure_modes.pdf")
    fig.savefig(OUT / "fig7_failure_modes.png", dpi=200)
    plt.close(fig)
    print("  fig7_failure_modes")


# ================================================================ FIG 8
def fig8_threshold():
    """One threshold, two incompatible jobs.

    The detector threshold was tuned to maximise top-box accuracy. Any program that
    must SELECT among candidates wants a much smaller set. The same knob cannot
    serve both, which is why depth ordering and per-candidate attribute checks both
    struggle at the tuned value.
    """
    f = Path("results/threshold_sweep.json")
    if not f.exists():
        print("  fig8 skipped (no threshold_sweep.json)")
        return
    d = json.load(open(f))
    rows = sorted(d["rows"], key=lambda r: r["threshold"])
    thr = [r["threshold"] for r in rows]
    acc_ = [r["acc50"] for r in rows]
    box = [r["mean_boxes"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(thr, acc_, "o-", color=NAVY, lw=2, ms=5,
            label="top-box accuracy (detection)")
    ax.set_xlabel("GLIP confidence threshold")
    ax.set_ylabel("top-box accuracy, IoU $\\\\geq$ 0.5 (%)", color=NAVY)
    ax.tick_params(axis="y", labelcolor=NAVY)
    ax.set_ylim(0, 95)

    ax2 = ax.twinx()
    ax2.plot(thr, box, "s--", color=GOLD, lw=1.8, ms=5,
             label="candidates per image")
    ax2.set_ylabel("candidates returned per image", color=GOLD)
    ax2.tick_params(axis="y", labelcolor=GOLD)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(box) * 1.15)

    ax.axvline(0.2, color=GREY, lw=1.0, ls=":")
    ax.annotate("chosen: 0.2\\nbest for detection\\n(≈20 candidates)",
                xy=(0.2, 12), xytext=(0.28, 26), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
    ax.annotate("selection wants\\nfar fewer", xy=(0.45, 62), xytext=(0.36, 74),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig)
    fig.savefig(OUT / "fig8_threshold_tradeoff.pdf")
    fig.savefig(OUT / "fig8_threshold_tradeoff.png", dpi=200)
    plt.close(fig)
    print("  fig8_threshold_tradeoff")


# ================================================================ FIG 9
def fig9_depth_sign():
    """Evidence for the depth sign, on ground-truth boxes with no detector.

    Kept as a figure because everything in Milestone 3 rests on it, and because 8%
    being BELOW chance — not merely low — is what identified the fault as an
    inversion rather than a weak signal.
    """
    labels = ["raw MiDaS output\\n(larger = closer)", "chance\\n(3.8 objects/image)",
              "inverted\\n(larger = further)"]
    vals = [8.0, 26.0, 80.0]
    colours = [RUST, GREY, TEAL]

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    bars = ax.bar(labels, vals, 0.55, color=colours)
    annotate(ax, bars, "{:.0f}%", dy=1.5)
    ax.axhline(26, color=GREY, lw=1.0, ls="--")
    ax.set_ylabel("target at correct depth extreme (%)")
    ax.set_ylim(0, 95)
    ax.set_title("RefCOCO+ depth queries, ground-truth boxes, n=25",
                 fontsize=8.5, color="#444444", pad=6)
    ax.text(0.02, 27.5, "below chance ⇒ inverted, not weak",
            fontsize=7, color="#666666")
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig, reserve=0.1)
    fig.savefig(OUT / "fig9_depth_sign.pdf")
    fig.savefig(OUT / "fig9_depth_sign.png", dpi=200)
    plt.close(fig)
    print("  fig9_depth_sign")

'''

s = s.replace("\ndef main():", new_figs + "\ndef main():")
s = s.replace("    fig6_seeds()\n",
              "    fig6_seeds()\n    fig7_failure_modes()\n    fig8_threshold()\n    fig9_depth_sign()\n")
p.write_text(s)
print("added fig7, fig8, fig9")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
