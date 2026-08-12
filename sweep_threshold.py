"""GLIP threshold sweep on a larger sample than D8's 20.

D8 fixed threshold=0.2 from 20 RefCOCO/testA samples. That is too few to fix a
parameter that moves mean IoU from 0.491 to 0.776. This re-runs the sweep on a
held-out slice: samples 200-399, disjoint from the first 20 and from the 0-199
range used elsewhere, so the threshold is not tuned on the data it is reported on.
"""
import argparse, glob, json, os, pickle, sys
import numpy as np, torch
from PIL import Image

STORE = os.path.expanduser("~/hpc-prog/humayun/vipergpt_store")
DATA = STORE + "/data"
sys.path.insert(0, "src")

class _D(dict):
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., ix2-ix1) * max(0., iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.

def load(n, start):
    vdir = DATA + "/refcoco/refcoco"
    f = sorted(glob.glob(vdir + "/refs(*).p"),
               key=lambda x: ("unc" not in os.path.basename(x), x))[0]
    refs = pickle.load(open(f, "rb"))
    inst = json.load(open(vdir + "/instances.json"))
    ann = {a["id"]: a for a in inst["annotations"]}
    img = {i["id"]: i for i in inst["images"]}
    out = []
    for r in refs:
        if r.get("split") != "testA": continue
        a, i = ann.get(r["ann_id"]), img.get(r["image_id"])
        x, y, w, h = a["bbox"]
        out.append({"file": i["file_name"],
                    "query": r["sentences"][0]["sent"].strip(),
                    "gt": (x, y, x+w, y+h)})
    return out[start:start+n]

def noun(q):
    w = [t for t in q.replace(",", " ").split() if t.isalpha()]
    return w[-1] if w else q

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--start", type=int, default=200)
ap.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5")
a = ap.parse_args()

from vipergpt_repro.models.glip import GLIPModel
samples = load(a.n, a.start)
print(f"samples {a.start}..{a.start+a.n-1}  (n={len(samples)})")
print(f"CUDA: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no GPU"

rows = []
for thr in [float(t) for t in a.thresholds.split(",")]:
    cfg = _D({"paths": {"pretrained_models": STORE + "/pretrained_models"},
              "detect_thresholds": {"glip": thr}, "crop_larger_margin": False,
              "ratio_box_area_to_image_area": 0.0, "device": "cuda:0"})
    m = GLIPModel(cfg); m._lazy_load()
    ious, empty, nbox = [], 0, []
    for s in samples:
        p = f"{DATA}/coco/train2014/{s['file']}"
        if not os.path.exists(p): continue
        pil = Image.open(p).convert("RGB"); W, H = pil.size
        chw = torch.from_numpy(np.asarray(pil).transpose(2,0,1).astype("float32"))
        boxes = m.find(chw, noun(s["query"]))
        nbox.append(len(boxes))
        if not boxes:
            empty += 1; ious.append(0.0); continue
        ious.append(max(iou((b[0], H-b[3], b[2], H-b[1]), s["gt"]) for b in boxes))
    r = {"threshold": thr, "n": len(ious),
         "mean_iou": round(float(np.mean(ious)), 4),
         "acc50": round(100*float(np.mean([i >= .5 for i in ious])), 2),
         "no_detection": empty,
         "mean_boxes": round(float(np.mean(nbox)), 2)}
    rows.append(r)
    print(f"  thr={thr:<5} mean_iou={r['mean_iou']:.4f}  acc50={r['acc50']:6.2f}%  "
          f"empty={empty:3d}  boxes/img={r['mean_boxes']}")
    del m; torch.cuda.empty_cache()

os.makedirs("results", exist_ok=True)
json.dump({"start": a.start, "n": a.n, "rows": rows},
          open("results/threshold_sweep.json", "w"), indent=2)
best = max(rows, key=lambda r: r["acc50"])
print(f"\nBEST by acc50: threshold={best['threshold']}  acc50={best['acc50']}%  "
      f"mean_iou={best['mean_iou']}")
print("written: results/threshold_sweep.json")
