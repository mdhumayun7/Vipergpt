"""Module bus: a single dispatch point between generated programs and the models.

`ModuleBus.call(name, *args)` routes an API call (find, simple_query, ...) to the
registered implementation. Two backends:

- **real**: lazily loads GLIP / X-VLM / BLIP-2 / MiDaS / text-LLM (GPU + weights).
- **mock**: deterministic CPU stand-ins so the smoke path and unit tests run with
  no GPU and no downloaded weights. Mocks are clearly marked and never used for
  reported numbers.

The official repo uses a producer-consumer multiprocessing bus; we start
single-process (correctness first) and can add batching later without changing the
program-facing API.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ModuleBus:
    def __init__(self, cfg=None, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self._handlers: dict[str, Callable] = {}
        self._loaded: dict[str, object] = {}
        if mock:
            self._register_mocks()
        else:
            self._register_real()

    def call(self, name: str, *args):
        if name not in self._handlers:
            raise KeyError(f"No handler registered for module call '{name}'")
        return self._handlers[name](*args)

    def register(self, name: str, fn: Callable) -> None:
        self._handlers[name] = fn

    # ------------------------------------------------------------------ mocks
    def _register_mocks(self):
        """Deterministic CPU mocks. NOT for reported numbers."""
        import numpy as np

        def _center_box(image):
            _, h, w = image.shape
            return [(int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))]

        self.register("find", lambda image, obj: _center_box(image))
        self.register("verify_property", lambda image, obj, prop: True)
        self.register("best_text_match", lambda image, opts, prefix=None: opts[0] if opts else "")
        self.register("best_image_match", lambda images, content: 0)
        self.register("simple_query", lambda image, q=None: "mock-answer")
        self.register("compute_depth", lambda image: float(np.median(np.asarray(image))))
        self.register("llm_query", lambda q, long_answer=True: "mock-llm-answer")
        self.register("select_answer", lambda info, question, options: 0)
        logger.warning("ModuleBus running with MOCK modules — results are NOT valid for reporting.")

    # ------------------------------------------------------------------ real
    def _register_real(self):
        """Lazily wire real models based on cfg.load_models. Imported on first use
        so that importing this module never requires torch/transformers/GPU."""
        from vipergpt_repro.models import (  # noqa: F401
            blip2, clip_vlm, codegen, depth, glip, text_llm, xvlm,
        )

        lm = self.cfg.load_models

        if lm.get("glip", False):
            g = glip.GLIPModel(self.cfg)
            self.register("find", g.find)
            self._loaded["glip"] = g
        if lm.get("xvlm", False):
            x = xvlm.XVLMModel(self.cfg)
            self.register("verify_property", x.verify_property)
            self.register("best_text_match", x.best_text_match)
            self.register("best_image_match", x.best_image_match)
            self._loaded["xvlm"] = x
        elif lm.get("clip", False):
            # DEVIATION D9: CLIP substitutes for X-VLM on verify_property /
            # best_text_match / best_image_match. See models/clip_vlm.py.
            c = clip_vlm.CLIPModel_(self.cfg)
            self.register("verify_property", c.verify_property)
            self.register("best_text_match", c.best_text_match)
            self.register("best_image_match", c.best_image_match)
            self._loaded["clip"] = c
            logger.warning("Using CLIP for verify_property/best_match (D9): X-VLM not built.")
        if lm.get("blip2", False):
            b = blip2.BLIP2Model(self.cfg)
            self.register("simple_query", b.simple_query)
            self._loaded["blip2"] = b
        if lm.get("depth", False):
            d = depth.DepthModel(self.cfg)
            self.register("compute_depth", d.compute_depth)
            self._loaded["depth"] = d
        if lm.get("llm_qa", False):
            t = text_llm.TextLLM(self.cfg)
            self.register("llm_query", t.llm_query)
            self.register("select_answer", t.select_answer)
            self._loaded["llm_qa"] = t

        # D10: neutral fallbacks for the text-answering calls we do not load
        # (BLIP-2-XXL and an 8B text LLM are out of scope for the grounding task).
        # Without these, a program calling simple_query/llm_query raises KeyError and
        # its failure is attributable to OUR configuration rather than to the program.
        # Returning "" lets such programs run to completion and fail, if they fail,
        # on the return type alone — which is what we are actually measuring.
        # Grounding cannot be answered by a string, so this concedes nothing.
        for _name in ("simple_query", "llm_query", "select_answer"):
            if _name not in self._handlers:
                self.register(_name, lambda *a, **k: "")
                logger.warning("D10: '%s' unhandled -> neutral empty-string fallback.", _name)
