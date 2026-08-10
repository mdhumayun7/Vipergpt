"""Toy end-to-end test: the whole M0 path runs in well under a minute, no GPU."""
from vipergpt_repro.pipeline.run_smoke import main


def test_smoke_runs_clean():
    assert main() == 0
