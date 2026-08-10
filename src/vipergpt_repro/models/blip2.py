"""BLIP-2 wrapper — backs `simple_query` (basic visual perception QA).

Uses HF `blip2-flan-t5-xxl` in 8-bit (config default), matching the official repo.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class BLIP2Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_type = str(cfg.blip2.model_type)  # e.g. blip2-flan-t5-xxl
        self.half = bool(cfg.blip2.get("half_precision", True))
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        name = f"Salesforce/{self.model_type}"
        logger.info("Loading BLIP-2: %s (8bit=%s)", name, self.half)
        self._processor = Blip2Processor.from_pretrained(name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            name,
            load_in_8bit=self.half,
            torch_dtype=torch.float16 if self.half else torch.float32,
            device_map="auto",
        )

    def simple_query(self, image, question: str | None = None) -> str:
        self._lazy_load()
        import numpy as np
        from PIL import Image

        if question is None:
            question = "What is this?"
        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        pil = Image.fromarray(arr.astype("uint8"))

        prompt = f"Question: {question} Answer:"
        inputs = self._processor(pil, text=prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=32)
        return self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
