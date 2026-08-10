"""MiDaS depth wrapper — backs `compute_depth` (median relative depth of a crop)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DepthModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._transform = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch

        logger.info("Loading MiDaS (DPT_Large) via torch.hub")
        self._model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self._model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._transform = transforms.dpt_transform

    def compute_depth(self, image) -> float:
        self._lazy_load()
        import numpy as np
        import torch

        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        inp = self._transform(arr.astype("uint8"))
        with torch.no_grad():
            pred = self._model(inp)
        # MiDaS returns inverse depth; the official API returns median of the map.
        return float(pred.median().item())
