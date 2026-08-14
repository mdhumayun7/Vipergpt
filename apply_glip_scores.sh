#!/usr/bin/env bash
# GLIP wrapper: expose detection scores, and filter candidates by them.
#
# Measured motivation (results/nms_depth_ablation.json):
#   39.5 boxes/img -> depth_order 9.7%
#    5.8 boxes/img -> depth_order 22.6%
# Top-k filtering more than doubles depth-ordering accuracy. But the current top-k
# ranks by BOX AREA, because the wrapper throws the detector's scores away. Area is
# a poor confidence proxy and it costs oracle (83.9% -> 61.3%): large background
# boxes survive while small correct ones are cut.
#
# GLIP's BoxList carries a "scores" field. This exposes it and adds score-based
# top-k, keeping the default behaviour unchanged so existing results stay valid.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path

p = Path("src/vipergpt_repro/models/glip.py")
s = p.read_text()

# ---------------------------------------------------------------- 1. config knobs
old = '''        self.min_area_ratio = float(cfg.get("ratio_box_area_to_image_area", 0.0))
        self.device = cfg.get("device", "cuda:0")'''
new = '''        self.min_area_ratio = float(cfg.get("ratio_box_area_to_image_area", 0.0))
        self.device = cfg.get("device", "cuda:0")
        # Keep at most this many detections, ranked by the detector's own confidence.
        # 0 disables filtering, which is the behaviour every result before
        # 2026-08-14 was produced under. See docs/deviations.md D11.
        self.max_detections = int(cfg.get("max_detections", 0))
        self.nms_iou = float(cfg.get("find_nms_iou", 0.0))  # 0 disables'''
assert old in s, "config anchor not found"
s = s.replace(old, new)

# ---------------------------------------------------------------- 2. keep scores
old = '''        boxes = out.bbox.detach().cpu().numpy()  # (x1, y1, x2, y2), top-left origin'''
new = '''        boxes = out.bbox.detach().cpu().numpy()  # (x1, y1, x2, y2), top-left origin
        # The detector's own confidence. Ranking by this rather than by box area is
        # the difference between discarding duplicates and discarding small correct
        # detections.
        try:
            scores = out.get_field("scores").detach().cpu().numpy()
        except Exception:  # noqa: BLE001 - field name differs across GLIP variants
            scores = None
        if scores is None or len(scores) != len(boxes):
            scores = [0.0] * len(boxes)'''
assert old in s, "boxes anchor not found"
s = s.replace(old, new)

# ---------------------------------------------------------------- 3. carry scores
old = '''        result = []
        for x1, y1, x2, y2 in boxes:'''
new = '''        scored = []
        for (x1, y1, x2, y2), sc in zip(boxes, scores):'''
assert old in s, "loop anchor not found"
s = s.replace(old, new)

old = '''            # top-left origin -> bottom-left origin
            result.append((int(x1), int(height - y2), int(x2), int(height - y1)))
        return result'''
new = '''            # top-left origin -> bottom-left origin
            scored.append(((int(x1), int(height - y2), int(x2), int(height - y1)),
                           float(sc)))

        # Highest confidence first, so any truncation keeps the best.
        scored.sort(key=lambda t: t[1], reverse=True)

        if self.nms_iou > 0:
            scored = _greedy_nms(scored, self.nms_iou)
        if self.max_detections > 0:
            scored = scored[: self.max_detections]

        self.last_scores = [sc for _, sc in scored]
        return [b for b, _ in scored]'''
assert old in s, "return anchor not found"
s = s.replace(old, new)

# ---------------------------------------------------------------- 4. nms helper
helper = '''

def _greedy_nms(scored, iou_thr):
    """Greedy NMS over (box, score) pairs, assumed already sorted by score.

    Boxes are (left, lower, right, upper) in bottom-left origin, but IoU is
    orientation-independent so no conversion is needed.
    """
    kept = []
    for box, sc in scored:
        if all(_iou(box, k) < iou_thr for k, _ in kept):
            kept.append((box, sc))
    return kept


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
'''
s = s.rstrip() + "\n" + helper
p.write_text(s)
print("[1/1] glip.py: scores exposed, score-ranked top-k and NMS added")
PYEOF

python -c "import ast; ast.parse(open('src/vipergpt_repro/models/glip.py').read())" && echo "SYNTAX OK"
echo
echo "Defaults unchanged (max_detections=0, find_nms_iou=0) — existing results stand."
echo "Enable per-experiment via cfg: max_detections=6, find_nms_iou=0.5"
