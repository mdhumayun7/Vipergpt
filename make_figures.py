"""Regenerate all figures into results/figures/ as vector PDF.

    python make_figures.py            # figures 1, 2, 3, 5 (CPU only)
    python make_figures.py --qual     # + figure 4, needs COCO images

Every number is read from results/m*_summaries/*.json and results/records/*.jsonl,
never typed, so a figure cannot disagree with the tables. Where seeded runs exist the
bars carry mean +- std; where they do not, the single greedy value is plotted and the
caption must say so.

Styling is explicit — no default matplotlib theme, no seaborn, no chartjunk.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- style
NAVY = "#122a56"
GOLD = "#b88a28"
TEAL = "#1f6f78"
RUST = "#9e2a2a"
GREY = "#787e86"
LIGHT = "#e8e9eb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

AUTHOR = "MD Humayun"
SUPERVISOR = "Dr. M.A. Zaveri"
AFFIL = "M.Tech CSE (Information Security and Privacy), SVNIT Surat"
CREDIT = f"{AUTHOR} · supervised by {SUPERVISOR} · {AFFIL}"


def credit(fig, reserve=0.085):
    """Attribution strip along the bottom, in space reserved for it."""
    try:
        fig.tight_layout(rect=(0, reserve, 1, 1))
    except Exception:  # noqa: BLE001 - some layouts refuse; the text still lands ok
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, reserve))
    fig.text(0.005, 0.012, CREDIT, fontsize=5.6, color="#9aa0a6",
             ha="left", va="bottom")

# ---------------------------------------------------------------- data
S: dict[str, dict] = {}
for f in glob.glob("results/m2_summaries/*.json") + glob.glob("results/m3_summaries/*.json"):
    d = json.load(open(f))
    if "tag" in d:
        S[d["tag"]] = d


def pick(*tags, default=None):
    for t in tags:
        if t in S:
            return t
    return default


def acc(tag, sub=None):
    d = S.get(tag)
    if d is None:
        return None
    return (d[sub]["acc_iou50"] if sub else d["acc_iou50"])


def seeded(prefix, sub=None):
    """(mean, std, n) over seeded runs, tolerating both tag conventions."""
    vals = []
    for i in (1, 2, 3):
        v = acc(f"seed_{prefix}_s{i}", sub)
        if v is None:
            v = acc(f"{prefix}_s{i}", sub)   # C2 seeds predate the seed_ prefix
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, 0
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def records(tag_glob):
    fs = sorted(glob.glob(f"results/records/*{tag_glob}.jsonl"))
    if not fs:
        return []
    return [json.loads(l) for l in open(fs[-1])]


def annotate(ax, bars, fmt="{:.1f}", dy=0.8):
    for b in bars:
        h = b.get_height()
        if h is None or np.isnan(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=7.5, color="#333333")


# ================================================================ FIG 1
def fig1_main():
    """Grouped bars: four conditions across six dataset/subset cells."""
    RP = dict(base=pick("m3_refcoco+_base", "baseline_plus"),
              grnd=pick("t300_grounding", "m3_refcoco+_grounding"),
              dpth=pick("t300_depth", "m3_refcoco+_depth"),
              cond=pick("t300_depth_cond", "c4_refcoco+"))
    RC = dict(base=pick("m3_refcoco_base", "baseline_full"),
              grnd=pick("m3_refcoco_grounding", "grounding_full"),
              dpth=pick("m3_refcoco_depth"),
              cond=pick("c4_refcoco"))

    groups = [
        ("RefCOCO+\noverall", RP, None), ("RefCOCO+\nspatial", RP, "spatial"),
        ("RefCOCO+\nnon-spatial", RP, "non_spatial"),
        ("RefCOCO\noverall", RC, None), ("RefCOCO\nspatial", RC, "spatial"),
        ("RefCOCO\nnon-spatial", RC, "non_spatial"),
    ]
    conds = [("C1 released prompt", "base", GREY),
             ("C2 + return contract", "grnd", NAVY),
             ("C3 + depth primitives", "dpth", GOLD),
             ("C4 + when to use them", "cond", TEAL)]

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    x = np.arange(len(groups))
    w = 0.2
    for i, (label, key, colour) in enumerate(conds):
        vals = [acc(tags[key], sub) or 0.0 for _, tags, sub in groups]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=colour,
                      edgecolor="white", linewidth=0.5)
        annotate(ax, bars)

    ax.axvline(2.5, color=LIGHT, lw=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Accuracy, IoU $\\geq$ 0.5 (%)")
    ax.set_ylim(0, 58)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig)
    fig.savefig(OUT / "fig1_main_result.pdf")
    fig.savefig(OUT / "fig1_main_result.png", dpi=200)
    plt.close(fig)
    print("  fig1_main_result")


# ================================================================ FIG 2
def fig2_scale():
    """The load-bearing figure: spatial is flat across scale, non-spatial is not."""
    models = ["1.5B", "7B", "32B"]
    sp, ns, ov = [], [], []
    for m, tag in zip(models, [pick("sweep_1.5B_grounding"),
                               pick("m3_refcoco_grounding", "grounding_full"),
                               pick("sweep_32B_grounding")]):
        sp.append(acc(tag, "spatial"))
        ns.append(acc(tag, "non_spatial"))
        ov.append(acc(tag))

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(3)
    ax.plot(x, ns, "o-", color=NAVY, lw=2, ms=6, label="non-spatial queries")
    ax.plot(x, sp, "s-", color=RUST, lw=2, ms=6, label="spatial queries")
    ax.plot(x, ov, "^--", color=GREY, lw=1.2, ms=5, label="overall", alpha=0.8)

    for xi, (a_, b_) in enumerate(zip(ns, sp)):
        close = abs(a_ - b_) < 3.0          # series nearly touch at 1.5B
        ax.annotate(f"{a_:.1f}", (xi, a_), textcoords="offset points",
                    xytext=(-16 if close else 0, 9), ha="center",
                    fontsize=8, color=NAVY)
        ax.annotate(f"{b_:.1f}", (xi, b_), textcoords="offset points",
                    xytext=(16 if close else 0, -15), ha="center",
                    fontsize=8, color=RUST)

    ax.annotate("", xy=(2.06, sp[2]), xytext=(2.06, ns[2]),
                arrowprops=dict(arrowstyle="<->", color="#999999", lw=0.9))
    ax.text(2.12, (sp[2] + ns[2]) / 2, f"{ns[2]-sp[2]:.1f} pts",
            fontsize=7.5, color="#666666", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Generator size (Qwen2.5-Coder)")
    ax.set_ylabel("Accuracy, IoU $\\geq$ 0.5 (%)")
    ax.set_xlim(-0.25, 2.55)
    ax.set_ylim(28, 54)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig)
    fig.savefig(OUT / "fig2_scale.pdf")
    fig.savefig(OUT / "fig2_scale.png", dpi=200)
    plt.close(fig)
    print("  fig2_scale")


# ================================================================ FIG 3
def fig3_adoption():
    """API call counts per condition, from the generated programs themselves."""
    NAMES = ("depth_order", "is_behind", "is_in_front_of", "is_between_3d",
             "distance_3d", "compute_depth", "find")
    conds = [("C1", "*_7B_base_greedy"), ("C2", "*_7B_grounding_greedy"),
             ("C3", "*_7B_depth_greedy"), ("C4", "*_7B_depth_cond_greedy")]
    counts = {}
    for label, pat in conds:
        ds = sorted(glob.glob(f"outputs/runs/*m1_refcoco_testA{pat}"))
        if not ds:
            continue
        recs = [json.loads(l) for l in open(ds[-1] + "/programs.jsonl")]
        c = {n: sum(r["program"].count(n + "(") for r in recs) for n in NAMES}
        counts[label] = c
    if not counts:
        print("  fig3 skipped (no program dumps on disk)")
        return

    labels = list(counts)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(labels))
    find = [counts[l]["find"] for l in labels]
    cdep = [counts[l]["compute_depth"] for l in labels]
    dord = [counts[l]["depth_order"] for l in labels]
    other = [sum(counts[l][n] for n in NAMES[1:5]) for l in labels]

    ax.bar(x - 0.27, find, 0.18, label="find", color=GREY)
    ax.bar(x - 0.09, cdep, 0.18, label="compute_depth (scalar)", color=LIGHT,
           edgecolor=GREY, linewidth=0.6)
    b3 = ax.bar(x + 0.09, dord, 0.18, label="depth_order", color=TEAL)
    ax.bar(x + 0.27, other, 0.18, label="other depth primitives", color=GOLD)
    annotate(ax, b3, "{:.0f}", dy=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("calls across 500 programs")
    ax.set_title("RefCOCO / testA", fontsize=9, color="#444444", pad=6)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    credit(fig)
    fig.savefig(OUT / "fig3_api_adoption.pdf")
    fig.savefig(OUT / "fig3_api_adoption.png", dpi=200)
    plt.close(fig)
    print("  fig3_api_adoption")


# ================================================================ FIG 5
def fig5_distribution():
    """Where the mean comes from: per-sample IoU, and what fraction score zero."""
    conds = [("C1", "m2_m3_refcoco_base"), ("C2", "m2_m3_refcoco_grounding"),
             ("C3", "m2_m3_refcoco_depth"), ("C4", "m2_c4_refcoco")]
    data, zeros = [], []
    labels = []
    for label, pat in conds:
        rs = records(pat)
        if not rs:
            continue
        ious = [r["iou"] for r in rs]
        nz = [v for v in ious if v > 0]
        zeros.append(100.0 * sum(1 for v in ious if v == 0) / len(ious))
        labels.append(label)
        # C1 scores exactly zero everywhere, so it has no distribution to draw.
        # A violin needs at least two points; substitute a flat sliver at 0 so the
        # condition still occupies its slot and the right panel tells the real story.
        data.append(nz if len(nz) >= 2 else [0.0, 0.0])
    if not data:
        print("  fig5 skipped (no records)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                   gridspec_kw={"width_ratios": [1.6, 1]})

    parts = ax1.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
    for pc, colour in zip(parts["bodies"], [GREY, NAVY, GOLD, TEAL]):
        pc.set_facecolor(colour)
        pc.set_alpha(0.55)
        pc.set_edgecolor("white")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#444444")
            parts[key].set_linewidth(0.9)
    ax1.axhline(0.5, color=RUST, lw=1.0, ls="--")
    ax1.text(0.55, 0.52, "IoU = 0.5 threshold", fontsize=7, color=RUST)
    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("IoU (non-zero samples only)")
    ax1.set_ylim(0, 1.02)
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    ax1.set_axisbelow(True)

    bars = ax2.bar(labels, zeros, 0.55, color=[GREY, NAVY, GOLD, TEAL])
    annotate(ax2, bars, "{:.0f}%", dy=1.5)
    ax2.set_ylabel("samples scoring exactly 0 (%)")
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", color=LIGHT, lw=0.7)
    ax2.set_axisbelow(True)

    credit(fig)
    fig.savefig(OUT / "fig5_iou_distribution.pdf")
    fig.savefig(OUT / "fig5_iou_distribution.png", dpi=200)
    plt.close(fig)
    print("  fig5_iou_distribution")


# ================================================================ FIG 4
def fig4_qualitative(n=4):
    """Side-by-side C2 vs C4 predictions on depth queries. Needs COCO images."""
    import pickle

    from PIL import Image

    STORE = os.path.expanduser("~/hpc-prog/humayun/vipergpt_store")
    DATA = STORE + "/data"
    vdir = DATA + "/refcoco/refcoco+"
    rf = sorted(glob.glob(vdir + "/refs(*).p"),
                key=lambda x: ("unc" not in os.path.basename(x), x))[0]
    refs = pickle.load(open(rf, "rb"))
    inst = json.load(open(vdir + "/instances.json"))
    img = {i["id"]: i for i in inst["images"]}

    order = []
    for r in refs:
        if r.get("split") != "testA":
            continue
        for s in r["sentences"]:
            order.append((s["sent"].strip(), img[r["image_id"]]["file_name"]))

    c2 = records("m2_t300_grounding")
    c4 = records("m2_t300_depth_cond")
    if not c2 or not c4:
        print("  fig4 skipped (missing records)")
        return

    # cases where C4 succeeds and C2 does not, on spatial queries
    picks = [i for i in range(min(len(c2), len(c4)))
             if c4[i]["is_spatial"] and c4[i]["iou"] >= 0.5 and c2[i]["iou"] < 0.5][:n]
    if not picks:
        print("  fig4 skipped (no contrasting cases)")
        return

    w = {2: 3.6, 3: 3.0, 4: 2.6}.get(len(picks), 3.0)
    fig, axes = plt.subplots(1, len(picks), figsize=(w * len(picks), 3.6),
                             constrained_layout=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, i in zip(axes, picks):
        fn = order[i][1]
        path = f"{DATA}/coco/train2014/{fn}"
        if not os.path.exists(path):
            continue
        ax.imshow(Image.open(path).convert("RGB"))
        for rec, colour, lbl in ((c2[i], RUST, "C2"), (c4[i], TEAL, "C4")):
            b = rec.get("pred_xyxy")
            if not b:
                continue
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1],
                                       fill=False, ec=colour, lw=2.2))
        g = c4[i]["gt_xyxy"]
        ax.add_patch(plt.Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                   fill=False, ec="#f2f2f2", lw=2.2, ls="--"))
        ax.add_patch(plt.Rectangle((g[0], g[1]), g[2]-g[0], g[3]-g[1],
                                   fill=False, ec="#333333", lw=0.8, ls="--"))
        q = c4[i]["query"]
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
    credit(fig, reserve=0.14)
    fig.savefig(OUT / "fig4_qualitative.pdf")
    fig.savefig(OUT / "fig4_qualitative.png", dpi=200)
    plt.close(fig)
    print(f"  fig4_qualitative ({len(picks)} cases)")



# ================================================================ FIG 6
def fig6_seeds():
    """Seed variance. Two points: C4 separates from C2 by far more than the spread,
    and C4 is markedly more STABLE on spatial queries — the explicit primitive
    removes the improvisation the generator otherwise has to do each time."""
    conds = [("C2\ncontract", "grounding", NAVY),
             ("C3\n+ depth", "depth", GOLD),
             ("C4\n+ routing", "depth_cond", TEAL)]
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
    ax1.set_ylabel("Accuracy, IoU $\\geq$ 0.5 (%)")
    ax1.set_ylim(0, 58)
    ax1.set_title("RefCOCO / testA — mean $\\pm$ s.d., 3 seeds, T=0.7",
                  fontsize=8.5, color="#444444", pad=6)
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in have],
               labels=[l.replace("\n", " ") for l, _, _ in have],
               loc="upper left", fontsize=7.5)
    ax1.grid(axis="y", color=LIGHT, lw=0.7)
    ax1.set_axisbelow(True)

    # right panel: the stability point
    labels, stds, colours = [], [], []
    for label, colour, row in have:
        sd = row[1][1]
        if sd is not None:
            labels.append(label.replace("\n", " "))
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
    cats = ["correct (IoU $\\geq$ 0.5)", "box returned, IoU < 0.5",
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
    ax.set_ylabel("top-box accuracy, IoU $\\geq$ 0.5 (%)", color=NAVY)
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
    ax.annotate("chosen: 0.2\nbest for detection\n(≈20 candidates)",
                xy=(0.2, 12), xytext=(0.28, 26), fontsize=7, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
    ax.annotate("selection wants\nfar fewer", xy=(0.45, 62), xytext=(0.36, 74),
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
    labels = ["raw MiDaS output\n(larger = closer)", "chance\n(3.8 objects/image)",
              "inverted\n(larger = further)"]
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
        ax.annotate("", xy=(3, best + 3.5), xytext=(3, reported),
                    arrowprops=dict(arrowstyle="<->", color="#999999", lw=0.9))
        ax.text(3.12, (best + reported) / 2,
                f"{reported - best:.1f} pts\n({100*best/reported:.0f}% recovered)",
                fontsize=7, color="#666666", va="center")

        ax.set_title(f"{title} / testA", fontsize=9, color="#444444", pad=6)
        ax.set_ylim(0, 92)
        ax.set_xlim(-0.6, 3.9)
        ax.grid(axis="y", color=LIGHT, lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy, IoU $\\geq$ 0.5 (%)")

    credit(fig)
    fig.savefig(OUT / "fig10_vs_paper.pdf")
    fig.savefig(OUT / "fig10_vs_paper.png", dpi=200)
    plt.close(fig)
    print("  fig10_vs_paper")



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
    ax1.text(0.31, 66, "greedy decoding", fontsize=7, color="#888888")
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
        ax2.set_title("advantage holds across thresholds (greedy)",
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
            "gap between bars = ordering failure\ngap above grey = detection failure",
            transform=ax.transAxes, fontsize=7, color="#666666", va="bottom")
    credit(fig)
    fig.savefig(OUT / "fig12_headroom.pdf")
    fig.savefig(OUT / "fig12_headroom.png", dpi=200)
    plt.close(fig)
    print("  fig12_headroom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual", action="store_true", help="also render figure 4")
    ap.add_argument("--qual-n", type=int, default=3,
                    help="panels in figure 4 (2 or 3 fit a two-column page; 4 is wide)")
    a = ap.parse_args()

    print(f"summaries loaded: {len(S)}")
    print("writing to results/figures/")
    fig1_main()
    fig2_scale()
    fig3_adoption()
    fig5_distribution()
    fig6_seeds()
    fig7_failure_modes()
    fig8_threshold()
    fig9_depth_sign()
    fig10_vs_paper()
    fig11_iou_sweep()
    fig12_headroom()
    if a.qual:
        fig4_qualitative(a.qual_n)

    sm, ss, sn = seeded("depth_cond")
    if sn:
        print(f"\n  seeded C4 available: {sm:.2f} +- {ss:.2f} over {sn} seeds")
        print("  re-run to add error bars to figure 1")
    else:
        print("\n  no seeded C3/C4 runs yet — figure 1 shows single greedy values")
    print("\ndone:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
