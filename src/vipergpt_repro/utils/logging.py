"""Logging configured to write to both stdout and a run-local file.

Never use bare `print` in the pipeline — use a module logger so that SLURM
`.out`/`.err` files and the run directory log stay usable for later analysis.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(run_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Set up root logging. If `run_dir` is given, also write to run_dir/run.log."""
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (important under multiprocessing / repeated calls).
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return root
