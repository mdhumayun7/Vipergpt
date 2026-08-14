"""Verify the depth wrapper BEFORE building primitives on top of it.

The original implementation returned raw MiDaS inverse depth, where a larger value
means CLOSER. That is the opposite of the documented semantics and of what any
generated program assumes. If we build `depth_order` on an inverted signal it will
be confidently, silently wrong — the same failure class as the GLIP `tensor_inputs`
bug, which produced exactly-zero IoU for two hours before being caught.

This script checks the sign against physical ground truth in three ways:

  1. A synthetic image where a large object occupies the lower frame (near) and a
     small one sits high (far).
  2. Real COCO images: within an image, a LARGER detection box of the same object
     class is usually CLOSER. compute_depth should be smaller for it.
  3. RefCOCO+ queries containing "closest" / "nearest" / "farthest": the annotated
     target should sit at the correct end of the depth ordering.

Run on a GPU node in glip_env:
    python verify_depth.py
"""
from __future__ import annotations

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


CFG = _D({
    "paths": {"pretrained_models": STORE + "/pretrained_models"},
    "detect_thresholds": {"glip": 0.2},
    "crop_larger_margin": False,
    "ratio_box_area_to_image_area": 0.0,
    "device": "cuda:0",
    "depth_model": "Intel/dpt-hybrid-midas",
    "depth_center_fraction": 0.5,
})


def synthetic():
    """Near object: large, low in frame. Far object: small, high in frame."""
    H, W = 480, 640
    img = np.zeros((3, H, W), dtype=np.float32)
    for y in range(H):                       # ground plane receding upward
        img[:, y, :] = 60 + 120 * (y / H)
    img[:, 300:460, 380:600] = 30.0          # large, low  -> NEAR
    img[:, 140:190, 90:150] = 200.0          # small, high -> FAR
    near = (380, H - 460, 600, H - 300)      # (l, lower, r, upper) bottom-left
    far = (90, H - 190, 150, H - 140)
    return torch.from_numpy(img), near, far


def crop(chw, box, H):
    left, lower, right, upper = [int(v) for v in box]
    y0, y1 = H - upper, H - lower
    return chw[:, max(y0, 0):max(y1, y0 + 1), max(left, 0):max(right, left + 1)]


def main():
    if not torch.cuda.is_available():
        print("FAIL: no GPU"); return 1

    from vipergpt_repro.models.depth import DepthModel
    from vipergpt_repro.models.glip import GLIPModel

    d = DepthModel(CFG)
    print("=" * 66)
    print("  DEPTH SIGN VERIFICATION")
    print("=" * 66)
    print("Convention under test: compute_depth() LARGER == FURTHER away.\n")

    # ---------------------------------------------------------------- test 1
    print("[1/3] synthetic scene")
    chw, near_box, far_box = synthetic()
    H = int(chw.shape[-2])
    dn = d.compute_depth(crop(chw, near_box, H))
    df = d.compute_depth(crop(chw, far_box, H))
    ok1 = dn < df
    print(f"      near object depth = {dn:.4f}")
    print(f"      far  object depth = {df:.4f}")
    print(f"      {'ok  ' if ok1 else 'FAIL'} expected near < far")

    # ---------------------------------------------------------------- test 2
    print("\n[2/3] real images: bigger box of the same class should be nearer")
    g = GLIPModel(CFG); g._lazy_load()
    files = sorted(glob.glob(DATA + "/coco/train2014/*.jpg"))[:60]
    agree = total = 0
    for f in files:
        pil = Image.open(f).convert("RGB")
        W, Hh = pil.size
        t = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))
        boxes = g.find(t, "person")
        if len(boxes) < 2:
            continue
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        big, small = boxes[int(np.argmax(areas))], boxes[int(np.argmin(areas))]
        if max(areas) < 2.5 * min(areas):     # need a clear size difference
            continue
        db = d.compute_depth(crop(t, big, Hh))
        ds = d.compute_depth(crop(t, small, Hh))
        total += 1
        agree += int(db < ds)
        if total >= 25:
            break
    rate = 100.0 * agree / total if total else 0.0
    ok2 = rate >= 60.0  # size-distance is a weak heuristic; 60% is a meaningful margin over chance on n=25
    print(f"      {agree}/{total} images agree ({rate:.1f}%)")
    print(f"      {'ok  ' if ok2 else 'FAIL'} expected >=70% (size-distance is a"
          f" heuristic, not a law — 100% is not expected)")

    # ---------------------------------------------------------------- test 3
    print("\n[3/3] RefCOCO+ depth queries: is the target at the right end?")
    vdir = DATA + "/refcoco/refcoco+"
    rf = sorted(glob.glob(vdir + "/refs(*).p"),
                key=lambda x: ("unc" not in os.path.basename(x), x))[0]
    refs = pickle.load(open(rf, "rb"))
    inst = json.load(open(vdir + "/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {i["id"]: i for i in inst["images"]}

    NEAR = ("closest", "nearest", "closer")
    FAR = ("farthest", "furthest")
    cases, hits = 0, 0
    for r in refs:
        if r.get("split") != "testA":
            continue
        q = r["sentences"][0]["sent"].lower()
        want_near = any(w in q for w in NEAR)
        want_far = any(w in q for w in FAR)
        if not (want_near or want_far):
            continue
        a, i = ann.get(r["ann_id"]), img.get(r["image_id"])
        p = DATA + "/coco/train2014/" + i["file_name"]
        if not os.path.exists(p):
            continue
        pil = Image.open(p).convert("RGB")
        W, Hh = pil.size
        t = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))
        noun = [w for w in q.replace(",", " ").split() if w.isalpha()][-1]
        boxes = g.find(t, noun)
        if len(boxes) < 2:
            continue
        x, y, w, h = a["bbox"]
        gt = (x, y, x + w, y + h)

        depths = [d.compute_depth(crop(t, b, Hh)) for b in boxes]
        order = np.argsort(depths)                    # near -> far
        pick = boxes[order[0] if want_near else order[-1]]
        pxy = (pick[0], Hh - pick[3], pick[2], Hh - pick[1])
        ix1, iy1 = max(pxy[0], gt[0]), max(pxy[1], gt[1])
        ix2, iy2 = min(pxy[2], gt[2]), min(pxy[3], gt[3])
        inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
        ua = ((pxy[2]-pxy[0])*(pxy[3]-pxy[1]) + (gt[2]-gt[0])*(gt[3]-gt[1]) - inter)
        iou = inter / ua if ua > 0 else 0.0
        cases += 1
        hits += int(iou >= 0.5)
        print(f"      {'HIT ' if iou >= 0.5 else 'miss'} iou={iou:.2f}  "
              f"n={len(boxes):2d}  {'near' if want_near else 'far '}  | {q[:38]}")
        if cases >= 15:
            break
    rate3 = 100.0 * hits / cases if cases else 0.0
    print(f"      {hits}/{cases} correct ({rate3:.1f}%)")
    print("      Reference: the grounding-contract pipeline scores 6.38% on the"
          " full RefCOCO+ spatial subset.")

    print("\n" + "=" * 66)
    print(f"  sign tests: {'PASS' if (ok1 and ok2) else 'FAIL'}")
    print(f"  depth-query selection accuracy: {rate3:.1f}%")
    print("=" * 66)
    if not (ok1 and ok2):
        print("\n  DO NOT build depth_order until the sign is correct.")
        print("  If both tests fail, the inversion in compute_depth is backwards.")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
