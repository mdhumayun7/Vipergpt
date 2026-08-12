import glob, json, os, pickle, sys
import numpy as np, torch
from PIL import Image
STORE = os.path.expanduser("~/hpc-prog/humayun/vipergpt_store")
DATA = STORE + "/data"
sys.path.insert(0, "src")

class _D(dict):
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v

CFG = _D({"paths": {"pretrained_models": STORE + "/pretrained_models"},
          "detect_thresholds": {"glip": 0.5}, "crop_larger_margin": False,
          "ratio_box_area_to_image_area": 0.0, "device": "cuda:0"})

from vipergpt_repro.models.glip import GLIPModel

vdir = DATA + "/refcoco/refcoco"
rf = sorted(glob.glob(vdir + "/refs(*).p"), key=lambda f: ("unc" not in os.path.basename(f), f))[0]
refs = pickle.load(open(rf, "rb"))
inst = json.load(open(vdir + "/instances.json"))
ann = {a["id"]: a for a in inst["annotations"]}
imgs = {i["id"]: i for i in inst["images"]}
s = [r for r in refs if r.get("split") == "testA"][2]
info = imgs[s["image_id"]]
a = ann[s["ann_id"]]

pil = Image.open(DATA + "/coco/train2014/" + info["file_name"]).convert("RGB")
W, H = pil.size
print("image:", info["file_name"], "PIL (W,H):", (W, H))
print("instances.json (W,H):", (info.get("width"), info.get("height")))
x, y, w, h = a["bbox"]
print("GT xywh:", a["bbox"], "-> xyxy:", (x, y, x + w, y + h))

chw = torch.from_numpy(np.asarray(pil).transpose(2, 0, 1).astype("float32"))
print("tensor CHW:", tuple(chw.shape), "min/max:", float(chw.min()), float(chw.max()))

m = GLIPModel(CFG)
m._lazy_load()
raw = m._demo.inference(chw[[2, 1, 0]], "person")
print("RAW bbox:", raw.bbox.cpu().numpy().tolist()[:5])
print("RAW size field:", getattr(raw, "size", None))
print("wrapper out:", m.find(chw, "person")[:5])
