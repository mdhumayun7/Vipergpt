"""GLIP wrapper — backs `find` / `exists`.

Uses the official viper GLIP fork (updated CUDA kernels, further patched for sm_90;
see docs/deviations.md D5). Returns boxes in the (left, lower, right, upper)
BOTTOM-LEFT-origin convention used throughout this repo and by the RefCOCO evaluator.

This mirrors `OurGLIPDemo` in the official repo's vision_models.py. Five details
matter and were wrong in the first draft of this file:

1. The checkpoint lives under `GLIP/checkpoints/`, not `GLIP/`.
2. `MODEL.DEVICE` must be set on the config, and `cfg.num_gpus = 1`.
3. `GLIPDemo` applies its own `transforms`, which expect a CHW float tensor —
   NOT an HWC uint8 numpy array. Passing the latter silently produces garbage.
4. BGR conversion is a channel permute on the CHW tensor (`image[[2,1,0]]`),
   applied before `inference`, not a numpy reverse-slice.
5. Use the public `self.inference(...)`, not `compute_prediction` +
   the private `_post_process`.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class GLIPModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold = float(cfg.detect_thresholds.glip)
        self.crop_larger_margin = bool(cfg.get("crop_larger_margin", True))
        self.min_area_ratio = float(cfg.get("ratio_box_area_to_image_area", 0.0))
        self.device = cfg.get("device", "cuda:0")
        # Keep at most this many detections, ranked by the detector's own confidence.
        # 0 disables filtering, which is the behaviour every result before
        # 2026-08-14 was produced under. See docs/deviations.md D11.
        self.max_detections = int(cfg.get("max_detections", 0))
        self.nms_iou = float(cfg.get("find_nms_iou", 0.0))  # 0 disables
        self._demo = None

    # ------------------------------------------------------------------ loading
    def _lazy_load(self):
        if self._demo is not None:
            return

        import torch
        from maskrcnn_benchmark.config import cfg as glip_cfg  # type: ignore
        from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo  # type: ignore

        working_dir = os.path.join(str(self.cfg.paths.pretrained_models), "GLIP")
        config_file = os.path.join(working_dir, "configs", "glip_Swin_L.yaml")
        weight_file = os.path.join(working_dir, "checkpoints", "glip_large_model.pth")

        for p in (config_file, weight_file):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"GLIP asset missing: {p}\nRun: bash scripts/download_models.sh --full"
                )

        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file)
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", self.device])

        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        with torch.cuda.device(self.device):
            self._demo = GLIPDemo(
                glip_cfg,
                min_image_size=800,
                confidence_threshold=self.threshold,
                show_mask_heatmaps=False,
                # We pass CHW float tensors, not HWC numpy. Without this, GLIPDemo
                # takes original_image.shape[:-1] -> (3, H) and rescales boxes to a
                # 3-pixel-wide image, producing exactly-zero IoU everywhere.
                tensor_inputs=True,
            )
        logger.info(
            "GLIP loaded: %s (threshold=%.2f, device=%s)",
            os.path.basename(weight_file), self.threshold, self.device,
        )

    # ------------------------------------------------------------------ inference
    def find(self, image, object_name: str):
        """image: CHW float tensor (RGB). Returns [(left, lower, right, upper), ...]."""
        self._lazy_load()
        import torch

        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"find() expects a CHW 3-channel tensor, got {tuple(image.shape)}")

        # GLIPDemo's transforms expect the 0-255 range.
        img = image.float()
        if img.max() <= 1.0:
            img = img * 255.0

        # RGB -> BGR, as a channel permute on CHW (official: prepare_image).
        img = img[[2, 1, 0]]

        # Guard the pathological aspect-ratio case exactly as the official code does.
        ratio = img.shape[1] / img.shape[2]
        ratio = max(ratio, 1 / ratio)
        original_min = self._demo.min_image_size
        if ratio > 10:
            self._demo.min_image_size = int(original_min * 10 / ratio)
            self._demo.transforms = self._demo.build_transform()

        try:
            with torch.cuda.device(self.device):
                out = self._demo.inference(img, object_name)
        finally:
            if ratio > 10:
                self._demo.min_image_size = original_min
                self._demo.transforms = self._demo.build_transform()

        boxes = out.bbox.detach().cpu().numpy()  # (x1, y1, x2, y2), top-left origin
        # The detector's own confidence. Ranking by this rather than by box area is
        # the difference between discarding duplicates and discarding small correct
        # detections.
        try:
            scores = out.get_field("scores").detach().cpu().numpy()
        except Exception:  # noqa: BLE001 - field name differs across GLIP variants
            scores = None
        if scores is None or len(scores) != len(boxes):
            scores = [0.0] * len(boxes)
        height = int(img.shape[-2])
        width = int(img.shape[-1])
        area_img = height * width

        scored = []
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            if self.crop_larger_margin:
                mw, mh = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
                x1, y1, x2, y2 = x1 - mw, y1 - mh, x2 + mw, y2 + mh
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, width), min(y2, height)
            if (x2 - x1) * (y2 - y1) < self.min_area_ratio * area_img:
                continue
            # top-left origin -> bottom-left origin
            scored.append(((int(x1), int(height - y2), int(x2), int(height - y1)),
                           float(sc)))

        # Highest confidence first, so any truncation keeps the best.
        scored.sort(key=lambda t: t[1], reverse=True)

        if self.nms_iou > 0:
            scored = _greedy_nms(scored, self.nms_iou)
        if self.max_detections > 0:
            scored = scored[: self.max_detections]

        self.last_scores = [sc for _, sc in scored]
        return [b for b, _ in scored]

    def exists(self, image, object_name: str) -> bool:
        return len(self.find(image, object_name)) > 0


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
