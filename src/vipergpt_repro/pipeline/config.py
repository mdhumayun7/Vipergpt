"""Configuration loading and merging via OmegaConf.

Merge order: default.yaml <- experiment file(s) <- CLI dotlist overrides.
Mirrors the official viper behaviour (base_config <- named configs).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from vipergpt_repro.utils.paths import CONFIGS


def load_config(
    experiment: str | None = None,
    overrides: Sequence[str] | None = None,
    default_name: str = "default.yaml",
) -> DictConfig:
    """Load and merge configuration.

    Parameters
    ----------
    experiment : str | None
        Name (without extension) of a file in configs/experiment/, e.g.
        "repro_table1_refcoco". If None, only the default config is used.
    overrides : Sequence[str] | None
        OmegaConf dotlist overrides, e.g. ["dataset.batch_size=4"].
    """
    cfg = OmegaConf.load(CONFIGS / default_name)

    if experiment:
        exp_path = CONFIGS / "experiment" / f"{experiment}.yaml"
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {exp_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    return cfg  # type: ignore[return-value]


def config_to_container(cfg: DictConfig) -> dict:
    """Resolve to a plain dict (for run stamping / JSON serialization)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def find_repo_configs() -> list[Path]:
    return sorted((CONFIGS / "experiment").glob("*.yaml"))
