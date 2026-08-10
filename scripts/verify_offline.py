"""Prove offline model loading works BEFORE submitting any compute-node job.

Compute nodes have no internet. A job that silently tries to reach huggingface.co
will hang until it hits the SLURM time limit and waste the allocation. Run this on
the LOGIN node with the network deliberately disabled:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_offline.py

Exits non-zero if anything required is missing. Checks tokenizer + config + actual
weight shards, and by default instantiates the model on CPU (meta device) so a
truncated download is caught here rather than inside a queued job.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_MODELS = [
    ("codegen", "Qwen/Qwen2.5-Coder-7B-Instruct"),
]
OPTIONAL_MODELS = [
    ("depth", "Intel/dpt-hybrid-midas"),
]


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_env() -> bool:
    offline = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    print("== environment ==")
    if not offline:
        _fail("HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE not both set to 1 — "
              "this run may silently reach the network and prove nothing")
    else:
        _ok("offline mode active")
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    print(f"  HF_HOME = {hf_home}")
    if not Path(hf_home).expanduser().exists():
        _fail(f"HF_HOME does not exist: {hf_home}")
        return False
    return offline


def check_weight_files(repo_id: str) -> bool:
    """Confirm real weight shards exist — config.json alone proves nothing."""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception as exc:  # noqa: BLE001
        _fail(f"{repo_id}: not in local cache ({type(exc).__name__})")
        return False

    shards = [p for p in path.rglob("*") if p.suffix in {".safetensors", ".bin"}]
    shards = [p for p in shards if p.stat().st_size > 1_000_000]
    if not shards:
        _fail(f"{repo_id}: no weight shards >1MB under {path}")
        return False

    total_gb = sum(p.stat().st_size for p in shards) / 1e9
    _ok(f"{repo_id}: {len(shards)} shard(s), {total_gb:.1f} GB")
    return True


def check_loads(repo_id: str, load_weights: bool) -> bool:
    from transformers import AutoConfig, AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
        AutoConfig.from_pretrained(repo_id, local_files_only=True)
        _ok(f"{repo_id}: tokenizer + config load")
    except Exception as exc:  # noqa: BLE001
        _fail(f"{repo_id}: tokenizer/config — {type(exc).__name__}: {exc}")
        return False

    if not load_weights:
        return True

    try:
        import torch
        from transformers import AutoModelForCausalLM

        with torch.device("meta"):
            AutoModelForCausalLM.from_pretrained(
                repo_id, local_files_only=True, dtype=torch.bfloat16
            )
        _ok(f"{repo_id}: weights instantiate (meta device)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"{repo_id}: weight load — {type(exc).__name__}: {exc}")
        return False
    return True


def check_data() -> bool:
    data_root = Path(
        os.environ.get(
            "DATA_PATH", Path.home() / "hpc-prog/humayun/vipergpt_store/data"
        )
    )
    print("== datasets ==")
    all_ok = True
    for version in ("refcoco", "refcoco+"):
        inst = data_root / "refcoco" / version / "instances.json"
        if inst.exists():
            _ok(f"{version}: {inst.stat().st_size / 1e6:.0f} MB")
        else:
            _fail(f"{version}: missing {inst}")
            all_ok = False
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-weights", action="store_true",
                    help="check files only; do not instantiate the model")
    ap.add_argument("--optional", action="store_true",
                    help="also check Milestone 2 perception models")
    args = ap.parse_args()

    results = [check_env()]

    print("== required models ==")
    for _role, repo in REQUIRED_MODELS:
        results.append(check_weight_files(repo))
        results.append(check_loads(repo, load_weights=not args.skip_weights))

    if args.optional:
        print("== optional models ==")
        for _role, repo in OPTIONAL_MODELS:
            check_weight_files(repo)

    results.append(check_data())

    print()
    if all(results):
        print("VERIFY OFFLINE: PASS — safe to submit jobs")
        return 0
    print("VERIFY OFFLINE: FAIL — fix the above on the LOGIN node before sbatch")
    return 1


if __name__ == "__main__":
    sys.exit(main())