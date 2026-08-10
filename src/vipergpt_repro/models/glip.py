"""GLIP wrapper — backs `find` / `exists`.

Uses the official viper GLIP fork (updated CUDA kernels). Returns bounding boxes in
the (left, lower, right, upper) BOTTOM-LEFT-origin convention used everywhere in
this repo and by the RefCOCO evaluator.

# NOTE (OQ-8 / R1): GLIP must be built from the cvlab-columbia/viper GLIP fork.
# See scripts/download_models.sh and docs/cluster_notes.md. This wrapper isolates
# that dependency so a build failure is a localized, well-marked problem.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GLIPModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold = float(cfg.detect_thresholds.glip)
        self.crop_larger_margin = bool(cfg.get("crop_larger_margin", True))
        self.min_area_ratio = float(cfg.get("ratio_box_area_to_image_area", 0.0))
        self._predictor = None

    def _lazy_load(self):
        if self._predictor is not None:
            return
        # Imported here so the package import never hard-requires the GLIP build.
        from maskrcnn_benchmark.config import cfg as glip_cfg  # type: ignore
        from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo  # type: ignore

        model_dir = f"{self.cfg.paths.pretrained_models}/GLIP"
        glip_cfg.local_rank = 0
        glip_cfg.merge_from_file(f"{model_dir}/configs/glip_Swin_L.yaml")
        glip_cfg.merge_from_list(["MODEL.WEIGHT", f"{model_dir}/glip_large_model.pth"])
        self._predictor = GLIPDemo(glip_cfg, min_image_size=800, confidence_threshold=self.threshold)
        logger.info("GLIP loaded (threshold=%.2f)", self.threshold)

    def find(self, image, object_name: str):
        """Return list of (left, lower, right, upper) boxes, bottom-left origin."""
        self._lazy_load()
        import numpy as np

        # image: torch float[3,H,W] in [0,1] or [0,255]; GLIP expects HWC BGR uint8.
        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)  # HWC
        if arr.max() <= 1.0:
            arr = arr * 255.0
        img_bgr = arr[:, :, ::-1].astype("uint8")
        h = img_bgr.shape[0]

        predictions = self._predictor.compute_prediction(img_bgr, object_name)
        top = self._predictor._post_process(predictions, threshold=self.threshold)
        boxes = top.bbox.tolist()  # (x1, y1, x2, y2) top-left origin

        out = []
        area_img = img_bgr.shape[0] * img_bgr.shape[1]
        for (x1, y1, x2, y2) in boxes:
            if self.crop_larger_margin:
                mw, mh = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
                x1, y1, x2, y2 = x1 - mw, y1 - mh, x2 + mw, y2 + mh
            if (x2 - x1) * (y2 - y1) < self.min_area_ratio * area_img:
                continue
            # convert to bottom-left origin: lower = h - y2, upper = h - y1
            out.append((int(x1), int(h - y2), int(x2), int(h - y1)))
        return out
