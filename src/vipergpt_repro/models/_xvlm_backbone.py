"""X-VLM backbone loader.

X-VLM is not on PyPI/HF-hub as a turn-key model; the official viper repo bundles it
under `base_models/` and loads a manually-downloaded checkpoint. On the cluster,
either (a) vendor the official `base_models/xvlm` implementation here, or (b) point
`load_xvlm` at the downloaded checkpoint via the config path.

This module intentionally fails loud until wired against the real checkpoint, so a
missing backbone is an obvious, localized error rather than a silent wrong result.
"""
from __future__ import annotations

from pathlib import Path


class _XVLMScorer:
    """Wraps an X-VLM model exposing `.score(image, text) -> float` (higher = better match)."""

    def __init__(self, model):
        self._model = model

    def score(self, image, text: str) -> float:  # pragma: no cover - needs checkpoint
        raise NotImplementedError(
            "Wire X-VLM image-text scoring here against the loaded checkpoint. "
            "See official viper base_models/xvlm for the forward pass."
        )


def load_xvlm(model_dir: str | Path) -> _XVLMScorer:  # pragma: no cover - needs checkpoint
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"X-VLM checkpoint dir not found: {model_dir}. Run scripts/download_models.sh "
            "and set paths.pretrained_models in the config."
        )
    raise NotImplementedError(
        "X-VLM backbone not yet vendored. Copy the official viper base_models/xvlm "
        "implementation into this package, then construct and return _XVLMScorer(model)."
    )
