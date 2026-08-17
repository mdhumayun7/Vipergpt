#!/usr/bin/env bash
# Add author attribution to every figure, and a sixth figure for seed variance.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

# ---------------------------------------------------------------- attribution
old = '''OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)'''
new = '''OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

AUTHOR = "MD Humayun"
SUPERVISOR = "Dr. M.A. Zaveri"
AFFIL = "M.Tech CSE (Information Security and Privacy), SVNIT Surat"
CREDIT = f"{AUTHOR} · supervised by {SUPERVISOR} · {AFFIL}"


def credit(fig):
    """Attribution strip along the bottom of every figure."""
    fig.text(0.005, 0.005, CREDIT, fontsize=5.6, color="#9aa0a6",
             ha="left", va="bottom")'''
assert old in s, "OUT anchor not found"
s = s.replace(old, new)

# call credit() before every save
s = s.replace('    fig.savefig(OUT / "fig', '    credit(fig)\n    fig.savefig(OUT / "fig')
# the double-save lines would get two credit() calls; collapse them
s = s.replace('    credit(fig)\n    fig.savefig(OUT / "fig1_main_result.png"',
              '    fig.savefig(OUT / "fig1_main_result.png"')
s = s.replace('    credit(fig)\n    fig.savefig(OUT / "fig2_scale.png"',
              '    fig.savefig(OUT / "fig2_scale.png"')
s = s.replace('    credit(fig)\n    fig.savefig(OUT / "fig3_api_adoption.png"',
              '    fig.savefig(OUT / "fig3_api_adoption.png"')
s = s.replace('    credit(fig)\n    fig.savefig(OUT / "fig5_iou_distribution.png"',
              '    fig.savefig(OUT / "fig5_iou_distribution.png"')
s = s.replace('    credit(fig)\n    fig.savefig(OUT / "fig4_qualitative.png"',
              '    fig.savefig(OUT / "fig4_qualitative.png"')

# ---------------------------------------------------------------- figure 6
fig6 = '''

# ================================================================ FIG 6
def fig6_seeds():
    """Seed variance. Two points: C4 separates from C2 by far more than the spread,
    and C4 is markedly more STABLE on spatial queries — the explicit primitive
    removes the improvisation the generator otherwise has to do each time."""
    conds = [("C2\\ncontract", "grounding", NAVY),
             ("C3\\n+ depth", "depth", GOLD),
             ("C4\\n+ routing", "depth_cond", TEAL)]
    subsets = [(None, "overall"), ("spatial", "spatial"), ("non_spatial", "non-spatial")]

    have = []
    for label, prefix, colour in conds:
        row = []
        for sub, _ in subsets:
            m, sd, n = seeded(prefix, sub)
            row.append((m, sd, n))
        if any(v[0] is not None for v in row):
            have.append((label, colour, row))
    if len(have) < 2:
        print("  fig6 skipped (need at least two seeded conditions)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    x = np.arange(len(subsets))
    w = 0.26
    for i, (label, colour, row) in enumerate(have):
        means = [(v[0] if v[0] is not None else np.nan) for v in row]
        errs = [(v[1] if v[1] is not None else 0.0) for v in row]
        off = (i - (len(have) - 1) / 2) * w
        bars = ax1.bar(x + off, means, w, yerr=errs, capsize=3,
                       color=colour, edgecolor="white", linewidth=0.5,
                       error_kw=dict(ecolor="#444444", lw=0.9))
        for b, m_, e_ in zip(bars, means, errs):
            if not np.isnan(m_):
                ax1.text(b.get_x() + b.get_width() / 2, m_ + e_ + 0.7,
                         f"{m_:.1f}", ha="center", fontsize=7, color="#333333")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[1] for s in subsets])
    ax1.set_ylabel("Accuracy, IoU $\\\\geq$ 0.5 (%)")
    ax1.set_ylim(0, 58)
    ax1.set_title("RefCOCO / testA — mean $\\\\pm$ s.d., 3 seeds, T=0.7",
                  fontsize=8.5, color="#444444", pad=6)
    ax1.legend([b[0] for b in have], loc="upper left", fontsize=7.5,
               handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in have],
               labels=[l.replace("\\n", " ") for l, _, _ in have])
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    ax1.set_axisbelow(True)

    # right panel: the stability point
    labels, stds, colours = [], [], []
    for label, colour, row in have:
        sd = row[1][1]
        if sd is not None:
            labels.append(label.replace("\\n", " "))
            stds.append(sd)
            colours.append(colour)
    bars = ax2.bar(labels, stds, 0.5, color=colours)
    annotate(ax2, bars, "{:.2f}", dy=0.06)
    ax2.set_ylabel("s.d. across seeds (pts)")
    ax2.set_title("spatial-query stability", fontsize=8.5, color="#444444", pad=6)
    ax2.set_ylim(0, max(stds) * 1.35 if stds else 1)
    ax2.grid(axis="y", color=LIGHT, lw=0.7)
    ax2.set_axisbelow(True)

    credit(fig)
    fig.savefig(OUT / "fig6_seed_variance.pdf")
    fig.savefig(OUT / "fig6_seed_variance.png", dpi=200)
    plt.close(fig)
    print("  fig6_seed_variance")

'''
s = s.replace("\ndef main():", fig6 + "\ndef main():")
s = s.replace("    fig5_distribution()\n", "    fig5_distribution()\n    fig6_seeds()\n")
p.write_text(s)
print("patched: attribution + fig6_seed_variance")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
