"""Deterministic seeding across random, numpy, and torch.

Every run must call `set_seed` and log the value so results are reproducible.
"""
from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed all RNGs and (optionally) enable deterministic backends.

    Returns the seed so callers can log/stamp it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy always present in this project, but keep util standalone
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt-in deterministic algorithms; some ops will raise if not supported.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # older torch
                pass
    except ImportError:
        pass

    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
    return seed
