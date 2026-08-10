"""Central path resolution. No hardcoded absolute paths elsewhere in src/."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"                     # gitignored, symlink to scratch
CHECKPOINTS = REPO_ROOT / "checkpoints"       # gitignored, symlink to scratch
PRETRAINED = REPO_ROOT / "pretrained_models"  # gitignored, symlink to scratch
OUTPUTS = REPO_ROOT / "outputs"               # gitignored
RESULTS = REPO_ROOT / "results"               # committed
