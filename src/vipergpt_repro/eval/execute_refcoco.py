"""Execute saved programs against RefCOCO and score them.

This is the Milestone 2 counterpart to the Milestone 1 static analysis. It reads a
`programs.jsonl` produced by the codegen stage, runs each program through the real
ModuleBus (GLIP + CLIP + MiDaS), and scores the returned patch against the RefCOCO
ground-truth box.

Why a separate runner rather than run_batch.py: generation needs the `vipergpt` env
(torch 2.13) and execution needs `glip_env` (torch 2.1.2, required by the GLIP
fork). The two cannot coexist in one interpreter. Splitting at the JSONL boundary
also means the before/after prompt comparison runs the *identical* programs.

Usage (GPU node, glip_env):

    python -m vipergpt_repro.eval.execute_refcoco \\
        --programs outputs/runs/<baseline_run>/programs.jsonl \\
        --max-samples 100 --tag baseline

    python -m vipergpt_repro.eval.execute_refcoco \\
        --programs outputs/runs/<grounding_run>/programs.jsonl \\
        --max-samples 100 --tag grounding
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time

logger = logging.getLogger("execute_refcoco")


class _D(dict):
    """OmegaConf stand-in so this runs in glip_env without omegaconf."""
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v


# ------------------------------------------------------------------ ground truth
def load_refcoco(data_root, version, split):
    """Returns records in the SAME order the codegen stage used, so index joins."""
    vdir = os.path.join(data_root, "refcoco", version)
    ref_files = sorted(glob.glob(os.path.join(vdir, "refs(*).p")),
                       key=lambda f: ("unc" not in os.path.basename(f), f))
    if not ref_files:
        raise FileNotFoundError(f"no refs(*).p under {vdir}")
    with open(ref_files[0], "rb") as f:
        refs = pickle.load(f)
    with open(os.path.join(vdir, "instances.json")) as f:
        inst = json.load(f)
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {i["id"]: i for i in inst["images"]}

    out = []
    for r in refs:
        if r.get("split") != split:
            continue
        a, i = ann.get(r["ann_id"]), img.get(r["image_id"])
        for s in r["sentences"]:
            x, y, w, h = a["bbox"]
            out.append({
                "query": s["sent"].strip(),
                "file": i["file_name"],
                "gt_xyxy": (x, y, x + w, y + h),   # top-left origin
            })
    return out


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify(result):
    """What did the program actually return?"""
    if result is None:
        return "none"
    if hasattr(result, "left") and hasattr(result, "upper"):
        return "patch"
    if isinstance(result, str):
        return "str"
    if isinstance(result, bool):
        return "bool"
    if isinstance(result, (list, tuple)):
        return "list"
    return type(result).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", required=True)
    ap.add_argument("--version", default="refcoco")
    ap.add_argument("--split", default="testA")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--tag", default="exec")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--store", default=os.path.expanduser("~/hpc-prog/humayun/vipergpt_store"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    import numpy as np
    import torch
    from PIL import Image

    from vipergpt_repro.pipeline.executor import execute_program
    from vipergpt_repro.pipeline.vision_bus import ModuleBus

    data_root = os.path.join(args.store, "data")
    cfg = _D({
        "paths": {"pretrained_models": os.path.join(args.store, "pretrained_models")},
        "detect_thresholds": {"glip": 0.2},        # D8
        "load_models": {"glip": True, "clip": True, "depth": True,
                        "xvlm": False, "blip2": False, "llm_qa": False},
        "crop_larger_margin": False,
        "ratio_box_area_to_image_area": 0.0,
        "device": "cuda:0",
        # D12: candidate-set controls, read from the environment so an ablation can
        # sweep them without touching code. Both default to 0 = disabled, which is
        # the configuration every previously reported number was produced under.
        "max_detections": int(os.environ.get("MAX_DET", "0")),
        "find_nms_iou": float(os.environ.get("FIND_NMS", "0")),
    })
    if cfg["max_detections"] or cfg["find_nms_iou"]:
        logger.info("D12 candidate cap active: max_detections=%s find_nms_iou=%s",
                    cfg["max_detections"], cfg["find_nms_iou"])

    # ---- programs
    with open(args.programs) as f:
        progs = [json.loads(line) for line in f]
    logger.info("loaded %d programs from %s", len(progs), args.programs)

    # ---- ground truth, joined by index with a query-equality assertion
    gt = load_refcoco(data_root, args.version, args.split)
    n = min(args.max_samples, len(progs), len(gt))
    mismatch = [i for i in range(n) if progs[i]["query"] != gt[i]["query"]]
    if mismatch:
        logger.error("query mismatch at indices %s — the programs were generated from a "
                     "different split/order. Refusing to score.", mismatch[:5])
        return 1
    logger.info("index join verified on %d samples", n)

    # ---- run dir
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            sha += "-dirty"
    except Exception:
        sha = "nogit"
    run_dir = os.path.join("outputs/runs",
                           f"{time.strftime('%Y%m%dT%H%M%S')}__{sha}__m2_{args.tag}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("run dir: %s", run_dir)

    bus = ModuleBus(cfg=cfg, mock=False)

    ious, kinds, errors = [], {}, {}
    recs = []
    t0 = time.time()
    for i in range(n):
        p, g = progs[i], gt[i]
        path = os.path.join(data_root, "coco", "train2014", g["file"])
        if not os.path.exists(path):
            logger.warning("missing image %s", g["file"])
            continue

        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        chw = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))

        res = execute_program(p["program"], chw, bus, timeout_s=args.timeout)
        kind = "error" if res.error else classify(res.result)
        kinds[kind] = kinds.get(kind, 0) + 1
        if res.error:
            key = res.error.split(":")[0]
            errors[key] = errors.get(key, 0) + 1

        iou = 0.0
        pred = None
        if kind == "patch":
            r = res.result
            # ImagePatch is bottom-left origin; undo the flip for COCO comparison.
            pred = (r.left, H - r.upper, r.right, H - r.lower)
            iou = iou_xyxy(pred, g["gt_xyxy"])
        ious.append(iou)   # non-patch returns score 0 — that is the point

        recs.append({
            "index": i, "query": p["query"], "is_spatial": p.get("is_spatial"),
            "kind": kind, "error": res.error, "pred_xyxy": pred,
            "gt_xyxy": g["gt_xyxy"], "iou": iou,
        })
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d  running mean IoU=%.3f  (%.1fs)",
                        i + 1, n, float(np.mean(ious)), time.time() - t0)

    with open(os.path.join(run_dir, "records.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    acc = 100.0 * float(np.mean([i >= 0.5 for i in ious])) if ious else 0.0
    sp = [r for r in recs if r.get("is_spatial")]
    nsp = [r for r in recs if not r.get("is_spatial")]

    summary = {
        "tag": args.tag, "programs": args.programs, "n": len(ious),
        "mean_iou": round(mean_iou, 4), "acc_iou50": round(acc, 2),
        "return_kinds": kinds, "errors": errors,
        "spatial": {
            "n": len(sp),
            "mean_iou": round(float(np.mean([r["iou"] for r in sp])), 4) if sp else 0.0,
            "acc_iou50": round(100.0 * float(np.mean([r["iou"] >= 0.5 for r in sp])), 2) if sp else 0.0,
        },
        "non_spatial": {
            "n": len(nsp),
            "mean_iou": round(float(np.mean([r["iou"] for r in nsp])), 4) if nsp else 0.0,
            "acc_iou50": round(100.0 * float(np.mean([r["iou"] >= 0.5 for r in nsp])), 2) if nsp else 0.0,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "git_sha": sha,
        "max_detections": cfg["max_detections"],
        "find_nms_iou": cfg["find_nms_iou"],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 64)
    print(f"  EXECUTION RESULT — {args.tag}")
    print("=" * 64)
    print(f"  samples          : {len(ious)}")
    print(f"  mean IoU         : {mean_iou:.4f}")
    print(f"  accuracy IoU>=0.5: {acc:.2f}%")
    print(f"  spatial     n={summary['spatial']['n']:<4} acc={summary['spatial']['acc_iou50']:.2f}%")
    print(f"  non-spatial n={summary['non_spatial']['n']:<4} acc={summary['non_spatial']['acc_iou50']:.2f}%")
    print(f"  return kinds     : {kinds}")
    if errors:
        print(f"  errors           : {errors}")
    print(f"  elapsed          : {summary['elapsed_s']}s")
    print(f"  written          : {run_dir}/summary.json")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
