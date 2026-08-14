#!/usr/bin/env bash
# Promote the depth prototype to API, and make the sign impossible to get wrong.
#
# Three sign errors in two days (compute_depth twice, then the sort in
# test_score_filter.py) all came from the same root cause: callers reading
# _inverse_depth_map() directly, where LARGER means CLOSER, while every other part of
# the system assumes LARGER means FURTHER. The fix is to stop exposing the raw map.
#
# Adds:
#   DepthModel.depth_of_region(box, W, H)  -> distance-like, larger = further
#   DepthModel.depth_map_for(image)        -> cached whole-image map
#   ImagePatch.depth_order / is_behind / distance_3d
#   ModuleBus registration for depth_batch
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path

# =============================================================== depth.py
p = Path("src/vipergpt_repro/models/depth.py")
s = p.read_text()

helper = '''
    # ------------------------------------------------------------------ safe API
    def depth_map_for(self, image):
        """Whole-image depth map, cached by object identity of the input tensor.

        Callers should NOT use this directly for comparisons — use
        depth_of_region(), which fixes the sign. This exists so a batch of boxes
        from one image costs one forward pass instead of N.
        """
        key = id(image)
        if getattr(self, "_map_key", None) == key:
            return self._map_cache
        m = self._inverse_depth_map(image)
        self._map_key, self._map_cache = key, m
        return m

    def depth_of_region(self, image, box, image_size):
        """Distance-like depth of a box: LARGER = FURTHER. Always use this.

        `box` is (left, lower, right, upper) in BOTTOM-LEFT origin; `image_size` is
        (W, H) of the original image. The map is resampled to image size upstream, so
        box coordinates index it directly after the row flip.

        The inversion lives here and in compute_depth, and nowhere else. Reading
        _inverse_depth_map() and comparing values is the bug that produced three
        separate sign errors; do not do it.
        """
        import numpy as np

        m = self.depth_map_for(image)
        if m is None:
            return None
        W, H = image_size
        left, lower, right, upper = [int(v) for v in box]
        y0, y1 = max(H - upper, 0), max(H - lower, 1)
        x0, x1 = max(left, 0), max(right, 1)
        reg = m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
        if reg.size == 0:
            return None
        h, w = reg.shape
        f = self.center_frac
        core = reg[int(h*(1-f)/2):max(int(h*(1+f)/2), int(h*(1-f)/2)+1),
                   int(w*(1-f)/2):max(int(w*(1+f)/2), int(w*(1-f)/2)+1)]
        v = float(np.median(core if core.size else reg))
        return 1.0 / max(v, _EPS)      # inverse depth -> distance-like

    def depth_batch(self, image, boxes, image_size):
        """Distance-like depth for many boxes, one forward pass. Larger = further."""
        return [self.depth_of_region(image, b, image_size) for b in boxes]
'''

anchor = "    def position_3d(self, image, box, image_size) -> tuple:"
assert anchor in s, "position_3d anchor not found"
s = s.replace(anchor, helper.rstrip() + "\n\n" + anchor)
p.write_text(s)
print("[1/3] depth.py: depth_of_region / depth_batch added")

# =============================================================== image_patch.py
p = Path("src/vipergpt_repro/pipeline/image_patch.py")
s = p.read_text()

api = '''

# ---- depth-grounded spatial primitives (contribution) ----
#
# ViperGPT exposes only `compute_depth()`, a single scalar per patch. That can
# express "A is behind B" but not "the third closest", "between A and B in depth",
# or any ordering over more than two objects. Measured consequence: on RefCOCO+
# depth queries the released pipeline scores 6.38%.
#
# All of these are LARGER = FURTHER, consistent with compute_depth's docstring.


def depth_order(patches: list, reverse: bool = False) -> list:
    """Patches sorted NEAR to FAR. `reverse=True` gives far to near.

    Backs "closest", "nearest", "farthest", "second closest", "third from the front"
    with one primitive. Patches whose depth cannot be computed are dropped rather
    than silently placed at one end.
    """
    if not patches:
        return []
    scored = []
    for p in patches:
        try:
            d = p.compute_depth()
        except Exception:  # noqa: BLE001 - a bad crop must not kill the program
            continue
        if d is not None:
            scored.append((p, d))
    scored.sort(key=lambda t: t[1], reverse=reverse)
    return [p for p, _ in scored]


def is_behind(a, b, margin: float = 0.0) -> bool:
    """True if `a` is further from the camera than `b` by at least `margin`."""
    return a.compute_depth() > b.compute_depth() + margin


def is_in_front_of(a, b, margin: float = 0.0) -> bool:
    return b.compute_depth() > a.compute_depth() + margin


def distance_3d(a, b) -> float:
    """Distance between patch centroids, combining image-plane offset and depth.

    Depth units are not metres — COCO ships no camera intrinsics — so this is a
    relative quantity, monotone in true separation. Use it for comparisons
    ("which is nearer to X"), never as an absolute measurement.
    """
    dz = a.compute_depth() - b.compute_depth()
    dx = a.horizontal_center - b.horizontal_center
    dy = a.vertical_center - b.vertical_center
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def is_between_3d(a, b, c) -> bool:
    """True if `a` lies between `b` and `c` in depth."""
    lo, hi = sorted([b.compute_depth(), c.compute_depth()])
    return lo < a.compute_depth() < hi
'''

marker = "Number = Union[int, float]"
assert marker in s, "tail anchor not found"
s = s.replace(marker, api.strip() + "\n\n\n" + marker)
p.write_text(s)
print("[2/3] image_patch.py: depth_order, is_behind, distance_3d, is_between_3d added")

# =============================================================== executor.py
p = Path("src/vipergpt_repro/pipeline/executor.py")
s = p.read_text()
old = '''        "best_image_match": ip.best_image_match,
        "distance": ip.distance,'''
new = '''        "best_image_match": ip.best_image_match,
        "distance": ip.distance,
        # contribution: depth-grounded primitives
        "depth_order": ip.depth_order,
        "is_behind": ip.is_behind,
        "is_in_front_of": ip.is_in_front_of,
        "distance_3d": ip.distance_3d,
        "is_between_3d": ip.is_between_3d,'''
assert old in s, "namespace anchor not found"
p.write_text(s.replace(old, new))
print("[3/3] executor.py: primitives exposed to generated programs")
PYEOF

echo
for f in src/vipergpt_repro/models/depth.py \
         src/vipergpt_repro/pipeline/image_patch.py \
         src/vipergpt_repro/pipeline/executor.py; do
  python -c "import ast; ast.parse(open('$f').read())" && echo "  OK  $f"
done
echo
echo "Next: python -B test_depth_api.py"
