"""Validate the GLIP wrapper against RefCOCO ground truth on REAL images.

The synthetic-image smoke test could not verify the coordinate convention, because
GLIP had no real object to find. This test does the decisive check: run GLIP on
actual COCO images with the referred object as the query, and compare against the
annotated box.

Interpretation:
  mean IoU > 0.4   -> the convention is correct; wiring is sound
  mean IoU ~ 0     -> boxes land nowhere near the target. Almost always a y-flip
                      error. The script re-scores WITHOUT the flip and reports
                      both, so the failure mode is unambiguous.

Run on a GPU node in glip_env:
    python test_glip_refcoco.py --n 20
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
DATA = f"{STORE}/data"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


class _D(dict):
    """Minimal OmegaConf stand-in: attribute access plus .get()."""
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v


CFG = _D({
    "paths": {"pretrained_models": f"{STORE}/pretrained_models"},
    "detect_thresholds": {"glip": float(__import__("os").environ.get("GLIP_THR", 0.5))},
    "crop_larger_margin": False,   # off: we are comparing against exact GT boxes
    "ratio_box_area_to_image_area": 0.0,
    "device": "cuda:0",
})


def iou_xyxy(a, b):
    """Standard IoU on (x1, y1, x2, y2), top-left origin."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_samples(version, split, n):
    vdir = f"{DATA}/refcoco/{version}"
    ref_files = sorted(glob.glob(f"{vdir}/refs(*).p"),
                       key=lambda f: ("unc" not in os.path.basename(f), f))
    with open(ref_files[0], "rb") as f:
        refs = pickle.load(f)
    with open(f"{vdir}/instances.json") as f:
        inst = json.load(f)

    ann = {a["id"]: a for a in inst["annotations"]}
    img = {i["id"]: i for i in inst["images"]}

    out = []
    for r in refs:
        if r.get("split") != split:
            continue
        a = ann.get(r["ann_id"])
        i = img.get(r["image_id"])
        if a is None or i is None:
            continue
        x, y, w, h = a["bbox"]                     # COCO: x, y, w, h top-left
        out.append({
            "file": i["file_name"],
            "query": r["sentences"][0]["sent"].strip(),
            "gt": (x, y, x + w, y + h),            # x1, y1, x2, y2 top-left
        })
        if len(out) >= n:
            break
    return out


def noun_of(query):
    """Crude head-noun guess: GLIP is a detector, not a grounding model, so the
    full referring expression ('the man on the left') is a poor detection prompt.
    Take the last word, which in RefCOCO is usually the object."""
    words = [w for w in query.replace(",", " ").split() if w.isalpha()]
    return words[-1] if words else query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--version", default="refcoco")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--use-full-query", action="store_true",
                    help="prompt GLIP with the whole referring expression")
    args = ap.parse_args()

    from vipergpt_repro.models.glip import GLIPModel

    print("=" * 66)
    print("  GLIP vs REFCOCO GROUND TRUTH")
    print("=" * 66)
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible.")
        return 1

    samples = load_samples(args.version, args.split, args.n)
    print(f"samples: {len(samples)}  ({args.version}/{args.split})")
    print(f"prompt  : {'full referring expression' if args.use_full_query else 'head noun only'}")

    model = GLIPModel(CFG)
    print("\nloading GLIP ...")
    model._lazy_load()
    print("ok\n")

    ious_flip, ious_noflip, empty = [], [], 0
    for k, s in enumerate(samples):
        path = f"{DATA}/coco/train2014/{s['file']}"
        if not os.path.exists(path):
            print(f"  [{k}] MISSING IMAGE {s['file']}")
            continue

        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        chw = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))

        prompt = s["query"] if args.use_full_query else noun_of(s["query"])
        boxes = model.find(chw, prompt)
        if not boxes:
            empty += 1
            print(f"  [{k}] '{s['query'][:34]:34s}' -> no detection")
            continue

        # As returned: (left, lower, right, upper) in BOTTOM-LEFT origin.
        # Undo the flip to get back to top-left for comparison with COCO GT.
        best_flip = max(iou_xyxy((b[0], H - b[3], b[2], H - b[1]), s["gt"]) for b in boxes)
        # Control: interpret the same numbers as if no flip had been applied.
        best_noflip = max(iou_xyxy((b[0], b[1], b[2], b[3]), s["gt"]) for b in boxes)

        ious_flip.append(best_flip)
        ious_noflip.append(best_noflip)
        print(f"  [{k}] '{s['query'][:34]:34s}' n={len(boxes):2d}  "
              f"IoU={best_flip:.3f}  (unflipped {best_noflip:.3f})")

    print("\n" + "=" * 66)
    if ious_flip:
        mf, mn = float(np.mean(ious_flip)), float(np.mean(ious_noflip))
        acc = 100.0 * float(np.mean([i >= 0.5 for i in ious_flip]))
        print(f"  scored           : {len(ious_flip)}   (no detection: {empty})")
        print(f"  mean IoU         : {mf:.3f}")
        print(f"  IoU>=0.5         : {acc:.1f}%")
        print(f"  mean IoU unflipped: {mn:.3f}   <- control")
        print()
        if mf > 0.4:
            print("  VERDICT: coordinate convention CORRECT.")
        elif mn > mf + 0.15:
            print("  VERDICT: Y-FLIP IS INVERTED — unflipped scores much better.")
            print("           Fix the height subtraction in models/glip.py.")
        else:
            print("  VERDICT: both low. Not a flip problem — check the prompt, the")
            print("           threshold, or image preprocessing.")
    else:
        print("  Nothing scored — GLIP returned no boxes at all.")
        print("  Try: --use-full-query, or lower detect_thresholds.glip.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
