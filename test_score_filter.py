"""Is the detector's confidence a better filter than box area?

Established (results/nms_depth_ablation.json), area-ranked top-k:

    filter      boxes/img   depth_order   oracle   recovered
    none            39.5          9.7%     83.9%       11.5%
    topk (area)      5.8         22.6%     61.3%       36.8%

Top-k more than doubles accuracy but costs 22 points of oracle: ranking by area
discards small correct detections and keeps large background ones. GLIP's own
confidence should rank better. This measures whether it does, at matched k.

Run on a GPU node in glip_env, AFTER apply_glip_scores.sh:
    python test_score_filter.py --n 40
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

STORE = os.path.expanduser("~/hpc-prog/humayun/vipergpt_store")
DATA = STORE + "/data"
sys.path.insert(0, "src")


class _D(dict):
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v


NEAR = ("closest", "nearest", "closer")
FAR = ("farthest", "furthest")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    it = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it / ua if ua > 0 else 0.0


def crop_region(full, box, W, H):
    """Median depth of a box read from ONE whole-image depth map.

    Relative depth is normalised per image, so values from separate per-crop
    inferences are not comparable. Reading every box from the same map is required
    for the ordering to mean anything.
    """
    fh, fw = full.shape
    left, lower, right, upper = box
    y0, y1 = int((H - upper) / H * fh), int((H - lower) / H * fh)
    x0, x1 = int(left / W * fw), int(right / W * fw)
    reg = full[max(y0, 0):max(y1, y0+1), max(x0, 0):max(x1, x0+1)]
    if reg.size == 0:
        return None
    h, w = reg.shape
    core = reg[int(h*0.25):max(int(h*0.75), int(h*0.25)+1),
               int(w*0.25):max(int(w*0.75), int(w*0.25)+1)]
    return float(np.median(core if core.size else reg))


def load(limit):
    vdir = DATA + "/refcoco/refcoco+"
    rf = sorted(glob.glob(vdir + "/refs(*).p"),
                key=lambda x: ("unc" not in os.path.basename(x), x))[0]
    refs = pickle.load(open(rf, "rb"))
    inst = json.load(open(vdir + "/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {i["id"]: i for i in inst["images"]}
    out = []
    for r in refs:
        if r.get("split") != "testA":
            continue
        q = r["sentences"][0]["sent"].lower()
        near, far = any(w in q for w in NEAR), any(w in q for w in FAR)
        if not (near or far):
            continue
        a, i = ann.get(r["ann_id"]), img.get(r["image_id"])
        p = DATA + "/coco/train2014/" + i["file_name"]
        if not os.path.exists(p):
            continue
        x, y, w, h = a["bbox"]
        out.append({"query": q, "path": p, "gt": (x, y, x+w, y+h), "near": near})
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("FAIL: no GPU"); return 1

    from vipergpt_repro.models.depth import DepthModel
    from vipergpt_repro.models.glip import GLIPModel

    base = {"paths": {"pretrained_models": STORE + "/pretrained_models"},
            "detect_thresholds": {"glip": 0.2}, "crop_larger_margin": False,
            "ratio_box_area_to_image_area": 0.0, "device": "cuda:0",
            "depth_model": "Intel/dpt-hybrid-midas", "depth_center_fraction": 0.5}

    g = GLIPModel(_D(dict(base)))
    g._lazy_load()
    d = DepthModel(_D(dict(base)))

    samples = load(a.n)
    print("=" * 72)
    print(f"  AREA vs CONFIDENCE RANKING   (k={a.k}, GLIP thr=0.2)")
    print("=" * 72)

    modes = ["all", "area_topk", "score_topk", "score_nms_topk"]
    st = {m: {"hit": 0, "orc": 0, "n": 0, "b": []} for m in modes}

    for s in samples:
        pil = Image.open(s["path"]).convert("RGB")
        W, H = pil.size
        t = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))
        noun = [w for w in s["query"].replace(",", " ").split() if w.isalpha()][-1]

        g.max_detections, g.nms_iou = 0, 0.0
        boxes = g.find(t, noun)
        scores = list(getattr(g, "last_scores", []))
        if len(boxes) < 2:
            continue
        if len(scores) != len(boxes):
            scores = [0.0] * len(boxes)

        full = d._inverse_depth_map(t)
        if full is None:
            continue

        area = lambda b: (b[2]-b[0]) * (b[3]-b[1])  # noqa: E731
        by_area = [boxes[i] for i in sorted(range(len(boxes)), key=lambda i: area(boxes[i]), reverse=True)][:a.k]
        by_score = [boxes[i] for i in sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)][:a.k]

        g.max_detections, g.nms_iou = a.k, 0.5
        by_score_nms = g.find(t, noun)
        g.max_detections, g.nms_iou = 0, 0.0

        for m, cand in (("all", boxes), ("area_topk", by_area),
                        ("score_topk", by_score), ("score_nms_topk", by_score_nms)):
            if len(cand) < 2:
                continue
            dep = [crop_region(full, b, W, H) for b in cand]
            pairs = [(b, v) for b, v in zip(cand, dep) if v is not None]
            if len(pairs) < 2:
                continue
            # crop_region returns RAW inverse depth (larger = CLOSER), unlike
            # DepthModel.compute_depth which inverts. Sorting ascending would put the
            # FURTHEST object first. Descending gives near -> far.
            pairs.sort(key=lambda t_: t_[1], reverse=True)
            pick = pairs[0][0] if s["near"] else pairs[-1][0]
            e = st[m]
            e["n"] += 1
            e["b"].append(len(cand))
            e["orc"] += int(max(iou((b[0], H-b[3], b[2], H-b[1]), s["gt"]) for b in cand) >= 0.5)
            e["hit"] += int(iou((pick[0], H-pick[3], pick[2], H-pick[1]), s["gt"]) >= 0.5)

    print(f"  {'ranking':<16}{'n':>4}{'boxes':>8}{'depth_order':>13}{'oracle':>9}{'recovered':>11}")
    print(f"  {'-'*16}{'-'*4}{'-'*8}{'-'*13}{'-'*9}{'-'*11}")
    out = {}
    for m in modes:
        e = st[m]
        if not e["n"]:
            print(f"  {m:<16}{0:>4}   (no usable cases)")
            continue
        acc = 100*e["hit"]/e["n"]; orc = 100*e["orc"]/e["n"]
        rec = 100*e["hit"]/e["orc"] if e["orc"] else 0.0
        out[m] = {"n": e["n"], "boxes": round(float(np.mean(e["b"])), 1),
                  "acc": round(acc, 1), "oracle": round(orc, 1), "recovered": round(rec, 1)}
        print(f"  {m:<16}{e['n']:>4}{np.mean(e['b']):>8.1f}{acc:>12.1f}%{orc:>8.1f}%{rec:>10.1f}%")

    print()
    print("  Compare against the area-ranked run: 22.6% acc, 61.3% oracle.")
    print("  If score ranking holds accuracy while raising oracle, it is strictly")
    print("  better and should replace area ranking in the contribution.")
    json.dump({"k": a.k, "results": out},
              open("results/score_filter_ablation.json", "w"), indent=2)
    print("\n  written: results/score_filter_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
