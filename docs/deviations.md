# Deviations from the paper

Living document. Every substitution away from what the paper specifies is
recorded here, in a `# DEVIATION:` code comment at the site, and in the README
results table. No silent substitutions.

| # | Area | Paper specifies | This repo uses | Reason | Likely effect on metric |
|---|------|-----------------|----------------|--------|-------------------------|
| D1 | Program generator (π) | OpenAI Codex `code-davinci-002` | Open-weights code model (finalized in plan) | Codex API deprecated 2023-03-23; unavailable | Uncertain; open code models generate less reliable programs → likely small negative |
| D2 | External-knowledge LLM (`llm_query`) | GPT-3 `text-davinci-003` | Open-weights instruct LLM | Cost + offline-cluster requirement; text-davinci-003 also deprecated | Possible negative on OK-VQA (knowledge recall) |

More rows added as implementation proceeds.

## D3 — RefCOCO annotation source
**Paper/official repo:** `https://bvisionweb1.cs.unc.edu/licheng/referit/data/`
**Used instead:** Internet Archive snapshots (20220413011718 / 20220413011656).
**Reason:** the canonical UNC host no longer resolves (verified 2026-08-10 from
two independent networks). Several published repos now use the same archived
zips. Contents are byte-identical to the originals.
**Effect on results:** none expected.

## D4 — Dependency versions
**environment.yml pins:** torch 2.1.*, pytorch-cuda 12.1, transformers 4.36.2
**Actually installed:** torch 2.13.0+cu130, transformers 5.15.0
**Reason:** the conda env pre-existed, so `conda env create` was a no-op and pip
resolved latest. Frozen to `requirements.lock.txt`.
**Effect on results:** none for the text-only Milestone 1 path. Material risk for
Milestone 2 — the GLIP fork's CUDA kernels predate sm_90.

## D5 — GLIP build for H100 (sm_90)
**Paper/official:** torch 1.13.1 + pytorch-cuda 11.6 (setup_env.sh). CUDA 11.6 has no
sm_90 support, so the published environment cannot run on H100 at all.
**Used instead:** torch 2.1.2+cu121, nvcc 12.8, TORCH_CUDA_ARCH_LIST="8.0;9.0".
**Source patches (7 files under maskrcnn_benchmark/csrc/cuda/):**
- `#include <THC/THCAtomics.cuh>` -> `<ATen/cuda/Atomic.cuh>`
- `#include <THC/THCDeviceUtils.cuh>` -> `<ATen/cuda/CUDAContext.h>`
  (THC/* was removed in torch 1.11+. THCCeilDiv/THCudaMalloc/THCState were not
  used in live code, so no further rewrites were needed.)
- setup.py: added `-gencode arch=compute_80,code=sm_80` and `compute_90/sm_90`.
**Build must run on a GPU node:** setup.py gates CUDAExtension on
`torch.cuda.is_available()`, so a login-node build silently produces a CPU-only
extension with no error.
**Backups:** *.cu.bak, setup.py.bak alongside the originals.
**Effect on results:** none expected; kernels are unchanged, only headers and target
architectures. To be confirmed by numerical equivalence on a fixed input.

## D6 — NumPy 2.0 aliases in GLIP
**Issue:** 5 files under maskrcnn_benchmark/ use `np.float`, removed in NumPy 1.24.
GLIP construction fails in anchor_generator.py before any inference.
**Fix:** np.float -> np.float64, np.int -> np.int64, np.bool -> bool, np.object -> object.
Backups as *.npbak. Semantics unchanged: np.float was always an alias for builtin
float, which numpy resolves to float64 in an array dtype.

## D7 — GLIP downloads a BERT tokenizer at load time
**Issue:** GLIPDemo pulls bert-base-uncased (440MB) from the Hub on first construction.
Compute nodes have no internet, so a SLURM job would hang until its time limit.
**Status:** now cached under HF_HOME. Job scripts MUST set HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1, and verify_offline.py should assert the BERT cache exists.
