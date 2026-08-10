"""Run-directory creation and stamping.

Each run gets a directory named:
    outputs/runs/<UTCstamp>__<gitsha>[-dirty]__<exp>/
containing a `run_stamp.json` with: git SHA, dirty flag, resolved config,
seed, environment lockfile hash, and hardware. A run from a dirty working
tree is tagged `-dirty` and MUST NOT be cited in the final results table.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def git_sha() -> str:
    return _git("rev-parse", "--short", "HEAD") or "nogit"


def is_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status)


def _lockfile_hash(repo_root: Path) -> str:
    lock = repo_root / "requirements.lock.txt"
    if not lock.exists():
        return "no-lockfile"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:12]


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return info


def create_run_dir(
    exp_name: str,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    repo_root: str | Path | None = None,
    outputs_root: str | Path | None = None,
) -> Path:
    """Create and stamp a run directory. Returns its path."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    outputs_root = Path(outputs_root) if outputs_root else repo_root / "outputs" / "runs"

    sha = git_sha()
    dirty = is_dirty()
    sha_tag = f"{sha}-dirty" if dirty else sha
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = outputs_root / f"{stamp}__{sha_tag}__{exp_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "exp_name": exp_name,
        "utc": stamp,
        "git_sha": sha,
        "dirty": dirty,
        "seed": seed,
        "lockfile_sha256": _lockfile_hash(repo_root),
        "hardware": _hardware(),
        "config": config or {},
    }
    (run_dir / "run_stamp.json").write_text(json.dumps(manifest, indent=2, default=str))

    if dirty:
        logger.warning(
            "Working tree is DIRTY. Run %s is tagged -dirty and cannot be cited in the results table.",
            run_dir.name,
        )
    logger.info("Created run dir: %s", run_dir)
    return run_dir
