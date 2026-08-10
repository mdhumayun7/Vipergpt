"""X-VLM wrapper — backs `verify_property`, `best_text_match`, `best_image_match`.

X-VLM is an image-text matching model. `verify_property` thresholds the match score
between the crop and "{object} that is {property}"; the best_* methods pick the
option / patch with the highest match score.

# NOTE (OQ-4): the paper text also mentions CLIP for best_*_match; the official
# config defaults to X-VLM. We follow the config (X-VLM) and record this in
# docs/deviations.md if it changes.

The X-VLM checkpoint is downloaded manually (see scripts/download_models.sh).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class XVLMModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.thresh = float(cfg.verify_property.thresh_xvlm)
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        # X-VLM loaded from the base_models implementation bundled with the official
        # repo. Isolated here; wiring finalized against the downloaded checkpoint.
        from vipergpt_repro.models._xvlm_backbone import load_xvlm  # provided in Phase 4/5

        model_dir = f"{self.cfg.paths.pretrained_models}/xvlm"
        self._model = load_xvlm(model_dir)
        logger.info("X-VLM loaded (verify threshold=%.2f)", self.thresh)

    def _score(self, image, text: str) -> float:
        self._lazy_load()
        return float(self._model.score(image, text))

    def verify_property(self, image, object_name: str, visual_property: str) -> bool:
        text = f"{object_name} that is {visual_property}"
        return self._score(image, text) > self.thresh

    def best_text_match(self, image, option_list, prefix: str | None = None) -> str:
        opts = [f"{prefix} {o}" if prefix else o for o in option_list]
        scores = [self._score(image, o) for o in opts]
        return option_list[int(max(range(len(scores)), key=lambda i: scores[i]))]

    def best_image_match(self, images, content) -> int:
        text = " ".join(content) if isinstance(content, (list, tuple)) else str(content)
        scores = [self._score(im, text) for im in images]
        return int(max(range(len(scores)), key=lambda i: scores[i]))
