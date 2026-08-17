#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("make_figures.py")
s = p.read_text()

# ---------------------------------------------------------------- fig 7
# Six categories under a horizontal bar chart collide with the x label. Put the
# legend down the right-hand side instead, where there is unused space.
old = '''    fig, ax = plt.subplots(figsize=(6.6, 3.2))'''
new = '''    fig, ax = plt.subplots(figsize=(8.2, 2.7))'''
assert old in s, "fig7 figsize"
s = s.replace(old, new)

old = '''    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig, reserve=0.24)'''
new = '''    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.015, 0.5),
              handlelength=1.1, labelspacing=0.6)
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig, reserve=0.12)'''
assert old in s, "fig7 legend"
s = s.replace(old, new)

# ---------------------------------------------------------------- fig 8
# The annotation drifted under the legend in the top-right corner.
old = '''    ax.annotate("selection wants\\\\nfar fewer", xy=(0.45, 62), xytext=(0.36, 74),
                fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))'''
new = '''    ax.annotate("selection wants\\\\nfar fewer candidates", xy=(0.42, 22),
                xytext=(0.30, 44), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))'''
assert old in s, "fig8 annotation"
s = s.replace(old, new)

# ---------------------------------------------------------------- fig 9
# The caption sat on top of the 26% bar label, and the y label was clipped.
old = '''    fig, ax = plt.subplots(figsize=(4.4, 2.9))'''
new = '''    fig, ax = plt.subplots(figsize=(4.8, 3.1))'''
assert old in s, "fig9 figsize"
s = s.replace(old, new)

old = '''    ax.set_ylabel("target at correct depth extreme (%)")
    ax.set_ylim(0, 95)
    ax.set_title("RefCOCO+ depth queries, ground-truth boxes, n=25",
                 fontsize=8.5, color="#444444", pad=6)
    ax.text(0.02, 27.5, "below chance ⇒ inverted, not weak",
            fontsize=7, color="#666666")'''
new = '''    ax.set_ylabel("target at correct\\ndepth extreme (%)", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_title("RefCOCO+ depth queries, ground-truth boxes, n=25",
                 fontsize=8.5, color="#444444", pad=6)
    ax.annotate("below chance ⇒\\ninverted, not weak", xy=(0, 9.5),
                xytext=(-0.34, 42), fontsize=7, color="#666666", ha="left",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))'''
assert old in s, "fig9 labels"
s = s.replace(old, new)

p.write_text(s)
print("patched fig7 / fig8 / fig9 layout")
PYEOF

python -c "import ast; ast.parse(open('make_figures.py').read())" && echo "SYNTAX OK"
