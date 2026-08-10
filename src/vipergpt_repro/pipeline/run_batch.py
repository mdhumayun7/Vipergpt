"""Batch reproduction runner.

    python -m vipergpt_repro.pipeline.run_batch experiment=repro_table1_refcoco

Loads a dataset, generates a program per sample with the code LLM, executes it
through the real ModuleBus, scores with the matching evaluator, and writes results
+ per-sample records into a stamped run directory.
"""
from __future__ import annotations

import argparse
import json
import logging

from vipergpt_repro.data import build_dataset, evaluate
from vipergpt_repro.models.codegen import build_codegen
from vipergpt_repro.pipeline.config import config_to_container, load_config
from vipergpt_repro.pipeline.executor import execute_program
from vipergpt_repro.pipeline.vision_bus import ModuleBus
from vipergpt_repro.utils.logging import configure_logging
from vipergpt_repro.utils.run import create_run_dir
from vipergpt_repro.utils.seed import set_seed

logger = logging.getLogger(__name__)


def parse_cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*", help="key=value overrides, incl. experiment=<name>")
    return ap.parse_args(argv)


def _split_overrides(overrides):
    experiment, rest = None, []
    for o in overrides:
        if o.startswith("experiment="):
            experiment = o.split("=", 1)[1]
        else:
            rest.append(o)
    return experiment, rest


def main(argv=None) -> int:
    args = parse_cli(argv)
    experiment, overrides = _split_overrides(args.overrides)

    cfg = load_config(experiment=experiment, overrides=overrides)
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = create_run_dir(experiment or "adhoc", config=config_to_container(cfg), seed=seed)
    configure_logging(run_dir)

    dataset = build_dataset(cfg)
    bus = ModuleBus(cfg=cfg, mock=False)
    gen = build_codegen(cfg, mock=False)

    records = []
    max_samples = cfg.dataset.get("max_samples")
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        query = sample["query"]
        program = gen.generate(query)
        image = sample["image"]
        exec_kwargs = sample.get("exec_kwargs", {})
        res = execute_program(program, image, bus, **exec_kwargs)
        records.append({
            "index": idx,
            "sample_id": sample.get("sample_id", idx),
            "query": query,
            "prediction": None if res.error else _serialize(res.result),
            "error": res.error,
            "ground_truth": sample.get("answer"),
            "program": program,
        })
        if idx % int(cfg.get("log_every", 20)) == 0:
            logger.info("processed %d samples", idx + 1)

    (run_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in records)
    )

    metrics = evaluate(cfg, records)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("METRICS %s", metrics)
    print(json.dumps({"run_dir": run_dir.name, "metrics": metrics}, indent=2))
    return 0


def _serialize(result):
    if hasattr(result, "left"):
        return [result.left, result.lower, result.right, result.upper]
    return result


if __name__ == "__main__":
    raise SystemExit(main())
