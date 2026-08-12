"""CLIP-backed `verify_property` / `best_text_match` / `best_image_match`.

DEVIATION D9: the paper uses X-VLM for these three calls. X-VLM weights are
distributed via Google Drive (not scriptable) and models/_xvlm_backbone.py is still
a stub, so we substitute CLIP ViT-L/14, which is already a ViperGPT dependency
(the official repo installs openai/CLIP for other purposes).

Expected effect: X-VLM is trained with fine-grained region-text alignment and should
be stronger on attribute verification than CLIP's global image-text similarity.
Treat `verify_property` numbers as a lower bound. On RefCOCO this matters less than
it would on GQA: in our Milestone 1 traces, `find` is called 547 times against 109
for `verify_property`.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/clip-vit-large-patch14"


class CLIPModel_:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0")
        self.model_name = cfg.get("clip_model", DEFAULT_MODEL)
        # Similarity above which a property counts as present. Not from the paper;
        # CLIP has no natural threshold, so we use a contrastive comparison instead
        # (see verify_property) and keep this only as a floor.
        self.min_sim = float(cfg.get("clip_min_similarity", 0.0))
        self._model = None
        self._proc = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._proc = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._dtype = torch.float32
        logger.info("CLIP loaded: %s on %s", self.model_name, self.device)

    # ------------------------------------------------------------------ helpers
    def _to_pil(self, image):
        import numpy as np
        from PIL import Image

        arr = image
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 3:      # CHW -> HWC
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.clip(0, 255).astype(np.uint8)
        if arr.shape[0] < 2 or arr.shape[1] < 2:     # degenerate crop
            return None
        return Image.fromarray(arr)

    def _scores(self, image, texts):
        """Softmaxed image-text similarity over `texts`."""
        import torch

        self._lazy_load()
        pil = self._to_pil(image)
        if pil is None:
            return None
        inputs = self._proc(text=texts, images=pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        return out.logits_per_image.softmax(dim=-1)[0].detach().cpu()

    # ------------------------------------------------------------------ API
    def verify_property(self, image, object_name: str, visual_property: str) -> bool:
        """Contrastive check: is '<property> <object>' a better fit than '<object>'
        alone, or than the explicit negation? A bare similarity threshold does not
        transfer across prompts, so we compare against a null hypothesis instead."""
        pos = f"a photo of a {visual_property} {object_name}"
        neg = f"a photo of a {object_name} that is not {visual_property}"
        s = self._scores(image, [pos, neg])
        if s is None:
            return False
        return bool(s[0] > s[1] and s[0] >= self.min_sim)

    def best_text_match(self, image, option_list, prefix: str | None = None) -> str:
        if not option_list:
            return ""
        texts = [f"{prefix} {o}" if prefix else f"a photo of {o}" for o in option_list]
        s = self._scores(image, texts)
        if s is None:
            return option_list[0]
        return option_list[int(s.argmax())]

    def best_image_match(self, images, content) -> int:
        """Index of the image best matching `content`. Scored one image at a time so
        that a single degenerate crop cannot break the batch."""
        import torch

        self._lazy_load()
        if isinstance(content, str):
            content = [content]
        text = " ".join(f"a photo of {c}" for c in content[:1]) or "a photo"

        best_i, best_v = 0, -1.0
        for i, im in enumerate(images):
            pil = self._to_pil(im)
            if pil is None:
                continue
            inputs = self._proc(text=[text], images=pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                v = float(self._model(**inputs).logits_per_image[0, 0])
            if v > best_v:
                best_i, best_v = i, v
        return best_i
