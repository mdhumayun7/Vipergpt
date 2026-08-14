"""Does candidate-set filtering recover depth ordering?

Established by the previous diagnostic:
  threshold 0.2 -> oracle 73%, depth_order 0%, 41.3 boxes/image
  threshold 0.7 -> oracle 75%, depth_order 50%,  2.2 boxes/image

So the correct box is usually present at 0.2; ordering 41 noisy candidates is what
fails. Raising the threshold fixes the ordering but throws away recall (many queries
return fewer than 2 boxes and are dropped entirely).

This tests a third option: keep threshold 0.2 for recall, then FILTER the candidate
set before ordering. Three filters, ablated separately so the contribution of each
is visible:

  none    - baseline, order all boxes (expected ~0%)
  nms     - greedy NMS at IoU 0.5, removing overlapping duplicates
  topk    - keep the k largest boxes (area is a crude confidence proxy since the
            wrapper does not currently return scores)
  nms+topk- both

Reports depth_order accuracy AND oracle for each, so a drop in accuracy caused by
filtering away the correct box is distinguishable from a failure to order.

GPU node, glip_env:
    python test_nms_depth.py --n 40
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


NEAR_WORDS = ("closest", "nearest", "closer")
FAR_WORDS = ("farthest", "furthest")


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, thr=0.5):
    """Greedy NMS by area (no scores available from the wrapper)."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)),
                   key=lambda i: (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]),
                   reverse=True)
    keep = []
    for i in order:
        if all(iou_xyxy(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep]


def topk(boxes, k=6):
    if len(boxes) <= k:
        return boxes
    order = sorted(range(len(boxes)),
                   key=lambda i: (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1]),
                   reverse=True)
    return [boxes[i] for i in order[:k]]


def crop(t, b, H):
    left, lower, right, upper = [int(v) for v in b]
    return t[:, max(H-upper, 0):max(H-lower, H-upper+1), max(left, 0):max(right, left+1)]


def load_depth_queries(limit):
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
        near = any(w in q for w in NEAR_WORDS)
        far = any(w in q for w in FAR_WORDS)
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
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no GPU"); return 1

    from vipergpt_repro.models.depth import DepthModel
    from vipergpt_repro.models.glip import GLIPModel

    cfg = _D({"paths": {"pretrained_models": STORE + "/pretrained_models"},
              "detect_thresholds": {"glip": a.threshold},
              "crop_larger_margin": False, "ratio_box_area_to_image_area": 0.0,
              "device": "cuda:0", "depth_model": "Intel/dpt-hybrid-midas",
              "depth_center_fraction": 0.5})

    g = GLIPModel(cfg); g._lazy_load()
    d = DepthModel(cfg)

    samples = load_depth_queries(a.n)
    print("=" * 70)
    print(f"  CANDIDATE FILTERING vs DEPTH ORDERING   (GLIP thr={a.threshold})")
    print("=" * 70)
    print(f"  depth queries: {len(samples)}   NMS IoU={a.nms_iou}   top-k={a.k}\n")

    filters = {
        "none":     lambda b: b,
        "nms":      lambda b: nms(b, a.nms_iou),
        "topk":     lambda b: topk(b, a.k),
        "nms+topk": lambda b: topk(nms(b, a.nms_iou), a.k),
    }
    stats = {k: {"hit": 0, "oracle": 0, "n": 0, "boxes": []} for k in filters}

    for s in samples:
        pil = Image.open(s["path"]).convert("RGB")
        W, H = pil.size
        t = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))
        noun = [w for w in s["query"].replace(",", " ").split() if w.isalpha()][-1]
        boxes = g.find(t, noun)
        if len(boxes) < 2:
            continue

        # depth is computed once per raw box and reused by every filter
        depth_of = {}
        for b in boxes:
            depth_of[b] = d.compute_depth(crop(t, b, H))

        for name, fn in filters.items():
            kept = fn(boxes)
            if len(kept) < 2:
                continue
            st = stats[name]
            st["n"] += 1
            st["boxes"].append(len(kept))
            st["oracle"] += int(max(
                iou_xyxy((b[0], H-b[3], b[2], H-b[1]), s["gt"]) for b in kept) >= 0.5)
            ordered = sorted(kept, key=lambda b: depth_of[b])
            pick = ordered[0] if s["near"] else ordered[-1]
            pxy = (pick[0], H-pick[3], pick[2], H-pick[1])
            st["hit"] += int(iou_xyxy(pxy, s["gt"]) >= 0.5)

    print(f"  {'filter':<10} {'n':>4} {'boxes/img':>10} {'depth_order':>12} {'oracle':>9} {'recovered':>10}")
    print(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*12} {'-'*9} {'-'*10}")
    out = {}
    for name in filters:
        st = stats[name]
        if st["n"] == 0:
            print(f"  {name:<10} {0:>4}  (no usable cases)")
            continue
        acc = 100.0 * st["hit"] / st["n"]
        orc = 100.0 * st["oracle"] / st["n"]
        rec = 100.0 * st["hit"] / st["oracle"] if st["oracle"] else 0.0
        mb = float(np.mean(st["boxes"]))
        out[name] = {"n": st["n"], "boxes_per_img": round(mb, 1),
                     "depth_order_acc": round(acc, 1), "oracle": round(orc, 1),
                     "recovered_pct": round(rec, 1)}
        print(f"  {name:<10} {st['n']:>4} {mb:>10.1f} {acc:>11.1f}% {orc:>8.1f}% {rec:>9.1f}%")

    print()
    print("  'recovered' = what fraction of the achievable (oracle) cases the depth")
    print("  ordering actually gets. It separates ordering failure from the correct")
    print("  box being absent or filtered away.")

    os.makedirs("results", exist_ok=True)
    json.dump({"threshold": a.threshold, "nms_iou": a.nms_iou, "k": a.k,
               "n_queries": len(samples), "results": out},
              open("results/nms_depth_ablation.json", "w"), indent=2)
    print("\n  written: results/nms_depth_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
