"""Smoke tests for the run-stamping and seeding utilities (no GPU needed)."""
from vipergpt_repro.utils.seed import set_seed


def test_set_seed_returns_seed():
    assert set_seed(123, deterministic=False) == 123


def test_set_seed_is_reproducible():
    import random

    set_seed(7, deterministic=False)
    a = [random.random() for _ in range(5)]
    set_seed(7, deterministic=False)
    b = [random.random() for _ in range(5)]
    assert a == b
