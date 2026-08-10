"""Toy end-to-end smoke run — CPU only, mock modules, no weights.

Proves the full path: prompt -> (mock) program generation -> compile -> execute
against a tiny synthetic image using (mock) modules. Runs in well under a minute.

    python -m vipergpt_repro.pipeline.run_smoke
"""
from __future__ import annotations

import logging

from vipergpt_repro.models.codegen import MockCodeGen
from vipergpt_repro.pipeline.executor import execute_program
from vipergpt_repro.pipeline.vision_bus import ModuleBus
from vipergpt_repro.utils.logging import configure_logging
from vipergpt_repro.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _toy_image():
    import numpy as np

    # channel-first float [3, H, W]
    return np.random.RandomState(0).rand(3, 64, 64).astype("float32")


def main() -> int:
    configure_logging()
    set_seed(0, deterministic=False)

    bus = ModuleBus(cfg=None, mock=True)
    gen = MockCodeGen()
    query = "What is this?"

    program = gen.generate(query)
    logger.info("Generated program:\n%s", program)

    image = _toy_image()
    res = execute_program(program, image, bus)

    if res.error:
        logger.error("SMOKE FAILED: %s", res.error)
        return 1
    logger.info("SMOKE OK — result: %r", res.result)
    print("SMOKE OK:", res.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
