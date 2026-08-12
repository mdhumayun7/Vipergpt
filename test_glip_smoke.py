"""Smoke test for the GLIP wrapper. Needs a GPU + the built extension, but NO
COCO images — it draws its own test image.

Run on a GPU node, in glip_env:

    python test_glip_smoke.py

Expected: GLIP loads, and `find("person")` on a photo-like synthetic image returns
zero or more boxes without raising. The strong check is the coordinate-convention
test at the end, which must produce a box in the LOWER half of the image.
"""
import os
import sys

import numpy as np
import torch

STORE = os.path.expanduser("~/hpc-prog/humayun/vipergpt_store")

class _D(dict):
    """Minimal stand-in for an OmegaConf node: attribute access + .get()."""
    def __getattr__(self, k):
        v = self[k]
        return _D(v) if isinstance(v, dict) else v

cfg = _D({
    "paths": {"pretrained_models": f"{STORE}/pretrained_models"},
    "detect_thresholds": {"glip": 0.5},
    "crop_larger_margin": True,
    "ratio_box_area_to_image_area": 0.0,
    "device": "cuda:0",
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from vipergpt_repro.models.glip import GLIPModel  # noqa: E402


def synthetic_image(h=480, w=640):
    """A crude scene: sky gradient, ground, and one dark blob in the LOWER half."""
    img = np.zeros((3, h, w), dtype=np.float32)
    for y in range(h):
        img[2, y, :] = 120 + 100 * (1 - y / h)   # blue sky fading down
        img[1, y, :] = 90 + 60 * (1 - y / h)
        img[0, y, :] = 70 + 40 * (1 - y / h)
    img[:, int(h * 0.7):, :] = np.array([90, 110, 70], dtype=np.float32)[:, None, None]
    # a vertical dark rectangle low in the frame
    y0, y1 = int(h * 0.55), int(h * 0.92)
    x0, x1 = int(w * 0.42), int(w * 0.56)
    img[:, y0:y1, x0:x1] = np.array([40, 40, 60], dtype=np.float32)[:, None, None]
    return torch.from_numpy(img)


def main():
    print("=" * 60)
    print("  GLIP WRAPPER SMOKE TEST")
    print("=" * 60)
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: no GPU. Get one with:")
        print("  srun --partition=gpu --gres=shard:25 --cpus-per-task=8 --mem=64G --pty bash")
        return 1
    print("gpu:", torch.cuda.get_device_name(0))

    print("\n[1/4] constructing wrapper")
    model = GLIPModel(cfg)
    print("      ok")

    print("\n[2/4] loading GLIP (slow the first time — 6.4GB checkpoint)")
    model._lazy_load()
    print("      ok")

    img = synthetic_image()
    print(f"\n[3/4] find('person') on synthetic image {tuple(img.shape)}")
    boxes = model.find(img, "person")
    print(f"      returned {len(boxes)} box(es): {boxes[:5]}")

    print("\n[4/4] coordinate convention check")
    h = int(img.shape[-2])
    ok = True
    for (left, lower, right, upper) in boxes:
        if not (0 <= left < right <= img.shape[-1]):
            print(f"      FAIL: bad x range ({left}, {right})")
            ok = False
        if not (0 <= lower < upper <= h):
            print(f"      FAIL: bad y range ({lower}, {upper}) — expected bottom-left origin")
            ok = False
    if boxes:
        # The blob sits low in the frame. In BOTTOM-LEFT origin that means a SMALL
        # `lower`. If `lower` is large for every box, the y-flip is inverted.
        if min(b[1] for b in boxes) > h * 0.5:
            print("      WARN: every box sits in the upper half in bottom-left coords,")
            print("            but the test blob was drawn low. Check the y-flip.")
        else:
            print("      ok — box positions consistent with bottom-left origin")
    else:
        print("      no boxes returned; convention untested.")
        print("      Not necessarily a failure: the synthetic image is not a real photo.")
        print("      Re-run against a real COCO image once train2014 has extracted.")

    print("\n" + "=" * 60)
    print("  RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
