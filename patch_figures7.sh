#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

fig13 = '''

# ================================================================ FIG 13
def fig13_cap_control():
    """The control that disqualified a 15-point result.

    Capping find() to three candidates raises C4 on RefCOCO from 42.8 to 57.6 — the
    largest single gain in the project. Applying the identical cap to C2, which has
    no depth primitives at all, raises it almost as much. The gain is therefore the
    candidate set, not the depth reasoning, and is not reportable as a contribution.

    Left panel: the two conditions under both settings. Right panel: the gain
    attributable to the cap, split by subset. Non-spatial barely moves, which is the
    mechanism — only "select one of N same-class objects" benefits from a smaller N,
    and it benefits regardless of how the selection is made.
    """
    need = ["ctl_grounding_k0", "ctl_depth_cond_k0",
            "ctl_grounding_k3", "ctl_depth_cond_k3"]
    if not all(t in S for t in need):
        print("  fig13 skipped (control runs not archived)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: absolute accuracy
    subsets = [(None, "overall"), ("spatial", "spatial"), ("non_spatial", "non-spatial")]
    x = np.arange(len(subsets))
    series = [("C2, no cap", "ctl_grounding_k0", "#7f93b5"),
              ("C2, k=3", "ctl_grounding_k3", NAVY),
              ("C4, no cap", "ctl_depth_cond_k0", "#8fbec2"),
              ("C4, k=3", "ctl_depth_cond_k3", TEAL)]
    w = 0.2
    for i, (label, tag, colour) in enumerate(series):
        vals = [acc(tag, sub) for sub, _ in subsets]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=label, color=colour,
                edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in subsets])
    ax1.set_ylabel("Accuracy, IoU $\\\\geq$ 0.5 (%)")
    ax1.set_ylim(0, 72)
    ax1.legend(fontsize=7, ncol=2, loc="upper center")
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.set_title("RefCOCO / testA", fontsize=8.5, color="#444444", pad=6)

    # ---- right: how much of the gain each condition gets from the cap alone
    gains_c2 = [acc("ctl_grounding_k3", sub) - acc("ctl_grounding_k0", sub)
                for sub, _ in subsets]
    gains_c4 = [acc("ctl_depth_cond_k3", sub) - acc("ctl_depth_cond_k0", sub)
                for sub, _ in subsets]
    b1 = ax2.bar(x - 0.18, gains_c2, 0.34, label="C2 (no depth primitives)", color=NAVY)
    b2 = ax2.bar(x + 0.18, gains_c4, 0.34, label="C4 (depth primitives)", color=TEAL)
    annotate(ax2, b1, "+{:.1f}", dy=0.5)
    annotate(ax2, b2, "+{:.1f}", dy=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in subsets])
    ax2.set_ylabel("gain from the cap (pts)")
    ax2.set_ylim(0, max(gains_c2 + gains_c4) * 1.32)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="y", color=LIGHT, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.set_title("almost identical $\\\\Rightarrow$ not attributable to depth",
                  fontsize=8.5, color="#444444", pad=6)

    credit(fig)
    fig.savefig(OUT / "fig13_cap_control.pdf")
    fig.savefig(OUT / "fig13_cap_control.png", dpi=200)
    plt.close(fig)
    print("  fig13_cap_control")

'''

s = s.replace("\ndef main():", fig13 + "\ndef main():")
s = s.replace("    fig12_headroom()\n", "    fig12_headroom()\n    fig13_cap_control()\n")
p.write_text(s)
print("added fig13_cap_control")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
