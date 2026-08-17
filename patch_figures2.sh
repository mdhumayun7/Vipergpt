#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

# ---------------------------------------------------------------- 1. attribution
# Reserve a strip at the bottom before writing the credit, otherwise it lands on
# top of the x tick labels.
old = '''def credit(fig):
    """Attribution strip along the bottom of every figure."""
    fig.text(0.005, 0.005, CREDIT, fontsize=5.6, color="#9aa0a6",
             ha="left", va="bottom")'''
new = '''def credit(fig, reserve=0.085):
    """Attribution strip along the bottom, in space reserved for it."""
    try:
        fig.tight_layout(rect=(0, reserve, 1, 1))
    except Exception:  # noqa: BLE001 - some layouts refuse; the text still lands ok
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, reserve))
    fig.text(0.005, 0.012, CREDIT, fontsize=5.6, color="#9aa0a6",
             ha="left", va="bottom")'''
assert old in s, "credit anchor"
s = s.replace(old, new)

# ---------------------------------------------------------------- 2. C2 seeds
# The C2 seed runs predate the seed_<prefix>_s<n> naming and are tagged
# grounding_s1..s3. Without this fallback figure 6 silently drops C2 — which is
# the condition the whole stability comparison is against.
old = '''def seeded(prefix, sub=None):
    """(mean, std, n) over seed_<prefix>_s1..s3 if present, else (None, None, 0)."""
    vals = [acc(f"seed_{prefix}_s{i}", sub) for i in (1, 2, 3)]
    vals = [v for v in vals if v is not None]'''
new = '''def seeded(prefix, sub=None):
    """(mean, std, n) over seeded runs, tolerating both tag conventions."""
    vals = []
    for i in (1, 2, 3):
        v = acc(f"seed_{prefix}_s{i}", sub)
        if v is None:
            v = acc(f"{prefix}_s{i}", sub)   # C2 seeds predate the seed_ prefix
        if v is not None:
            vals.append(v)'''
assert old in s, "seeded anchor"
s = s.replace(old, new)

# ---------------------------------------------------------------- 3. fig2 labels
# At 1.5B the two series nearly touch, so fixed offsets collide. Push them apart
# only where they are close.
old = '''    for xi, v in zip(x, ns):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=NAVY)
    for xi, v in zip(x, sp):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=RUST)'''
new = '''    for xi, (a_, b_) in enumerate(zip(ns, sp)):
        close = abs(a_ - b_) < 3.0          # series nearly touch at 1.5B
        ax.annotate(f"{a_:.1f}", (xi, a_), textcoords="offset points",
                    xytext=(-16 if close else 0, 9), ha="center",
                    fontsize=8, color=NAVY)
        ax.annotate(f"{b_:.1f}", (xi, b_), textcoords="offset points",
                    xytext=(16 if close else 0, -15), ha="center",
                    fontsize=8, color=RUST)'''
assert old in s, "fig2 label anchor"
s = s.replace(old, new)

# ---------------------------------------------------------------- 4. fig4 layout
# COCO images have different aspect ratios, so equal-width subplots left the panels
# at different heights with legends printed over the picture.
old = '''    w = {2: 3.6, 3: 2.9, 4: 2.5}.get(len(picks), 2.9)
    fig, axes = plt.subplots(1, len(picks), figsize=(w * len(picks), 3.4))'''
new = '''    w = {2: 3.6, 3: 3.0, 4: 2.6}.get(len(picks), 3.0)
    fig, axes = plt.subplots(1, len(picks), figsize=(w * len(picks), 3.6),
                             constrained_layout=True)'''
assert old in s, "fig4 figsize anchor"
s = s.replace(old, new)

old = '''        ax.set_title(f'"{c4[i]["query"][:34]}"', fontsize=8, color="#333333")
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.75)
        ax.axis("off")
    credit(fig)'''
new = '''        q = c4[i]["query"]
        ax.set_title(f'"{q[:30]}{"…" if len(q) > 30 else ""}"',
                     fontsize=8, color="#333333", pad=4)
        ax.set_anchor("N")          # top-align panels of differing aspect ratio
        ax.axis("off")

    # One shared legend below the row, so nothing is drawn over the images.
    handles = [plt.Line2D([], [], color=RUST, lw=2.2),
               plt.Line2D([], [], color=TEAL, lw=2.2),
               plt.Line2D([], [], color="#555555", lw=1.8, ls="--")]
    fig.legend(handles, ["C2 (return contract)", "C4 (+ depth routing)",
                         "ground truth"],
               loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    for ax, i in zip(axes, picks):
        ax.text(0.02, 0.02, f"C2 {c2[i]['iou']:.2f}   C4 {c4[i]['iou']:.2f}",
                transform=ax.transAxes, fontsize=7, color="white",
                va="bottom", ha="left",
                bbox=dict(facecolor="#00000099", edgecolor="none", pad=2))
    credit(fig, reserve=0.14)'''
assert old in s, "fig4 legend anchor"
s = s.replace(old, new)

# ground-truth box in white is invisible on pale images
s = s.replace('''                                   fill=False, ec="white", lw=1.6, ls="--",
                                   label="ground truth"))''',
              '''                                   fill=False, ec="#f2f2f2", lw=2.2, ls="--"))
        ax.add_patch(plt.Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                   fill=False, ec="#333333", lw=0.8, ls="--"))''')
# drop the now-unused per-patch labels
s = s.replace('''                                       fill=False, ec=colour, lw=2.0,
                                       label=f"{lbl} IoU {rec['iou']:.2f}"))''',
              '''                                       fill=False, ec=colour, lw=2.2))''')

p.write_text(s)
print("patched: attribution spacing, C2 seed fallback, fig2 labels, fig4 layout")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
