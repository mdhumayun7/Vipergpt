#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

fig10 = '''

# ================================================================ FIG 10
def fig10_vs_paper():
    """Reproduced against reported, with the measured perception ceiling.

    Deliberately NOT a waterfall. Attributing the gap to individual deviations
    (D1 Codex, D9 X-VLM, D8 threshold) would require per-deviation ablations we did
    not run; inventing that split would be fabrication. What is shown is measured:
    where each condition lands, what the paper reports, and where our own perception
    stack caps out.

    The ceiling is worth reading carefully. GLIP alone, taking its single best box,
    reaches 81.5% on RefCOCO (n=200 held out, threshold 0.2) — ABOVE the paper's
    72.0. So perception is not what separates us from the paper. The gap is program
    synthesis: Codex composed better programs over the same kind of detector than an
    open-weights model does.
    """
    RP = dict(base=pick("m3_refcoco+_base", "baseline_plus"),
              grnd=pick("t300_grounding", "m3_refcoco+_grounding"),
              dpth=pick("t300_depth", "m3_refcoco+_depth"),
              cond=pick("t300_depth_cond", "c4_refcoco+"))
    RC = dict(base=pick("m3_refcoco_base", "baseline_full"),
              grnd=pick("m3_refcoco_grounding", "grounding_full"),
              dpth=pick("m3_refcoco_depth"),
              cond=pick("c4_refcoco"))

    panels = [("RefCOCO", RC, 72.0, 81.5), ("RefCOCO+", RP, 67.0, None)]
    names = ["C1", "C2", "C3", "C4"]
    keys = ["base", "grnd", "dpth", "cond"]
    colours = [GREY, NAVY, GOLD, TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for ax, (title, tags, reported, ceiling) in zip(axes, panels):
        vals = [acc(tags[k]) or 0.0 for k in keys]
        bars = ax.bar(names, vals, 0.6, color=colours, edgecolor="white", linewidth=0.5)
        annotate(ax, bars, "{:.1f}", dy=1.0)

        ax.axhline(reported, color=RUST, lw=1.4, ls="--")
        ax.text(3.48, reported + 1.2, f"paper (Codex) {reported:.1f}",
                fontsize=7, color=RUST, ha="right")

        if ceiling is not None:
            ax.axhline(ceiling, color="#7a8b99", lw=1.1, ls=":")
            ax.text(3.48, ceiling + 1.2, f"GLIP top-box ceiling {ceiling:.1f}",
                    fontsize=7, color="#5c6b78", ha="right")

        best = vals[-1]
        ax.annotate("", xy=(3, best), xytext=(3, reported),
                    arrowprops=dict(arrowstyle="<->", color="#999999", lw=0.9))
        ax.text(3.12, (best + reported) / 2,
                f"{reported - best:.1f} pts\\n({100*best/reported:.0f}% recovered)",
                fontsize=7, color="#666666", va="center")

        ax.set_title(f"{title} / testA", fontsize=9, color="#444444", pad=6)
        ax.set_ylim(0, 92)
        ax.set_xlim(-0.6, 3.9)
        ax.grid(axis="y", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy, IoU $\\\\geq$ 0.5 (%)")

    credit(fig)
    fig.savefig(OUT / "fig10_vs_paper.pdf")
    fig.savefig(OUT / "fig10_vs_paper.png", dpi=200)
    plt.close(fig)
    print("  fig10_vs_paper")

'''

s = s.replace("\ndef main():", fig10 + "\ndef main():")
s = s.replace("    fig9_depth_sign()\n", "    fig9_depth_sign()\n    fig10_vs_paper()\n")
p.write_text(s)
print("added fig10_vs_paper")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
