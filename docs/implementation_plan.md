# Implementation Plan — ViperGPT reproduction

Derived from `docs/paper_spec.md`. Covers the module breakdown, reuse-vs-reimplement
decisions, pinned versions, config schema, milestones, compute budget, and risks.
ViperGPT is **training-free**, so "implementation" means faithfully wiring the
pretrained modules behind the API and getting the program generator + executor right.

---

## 1. Module breakdown (one module per pipeline stage)

| Stage | File under `src/vipergpt_repro/` | Build strategy | Justification |
|-------|----------------------------------|----------------|---------------|
| Config loading/merge | `pipeline/config.py` | **Reuse** OmegaConf | Matches official repo; standard, robust merging |
| Program generator `π` | `models/codegen.py` | **Reuse** HF `transformers` + open-weights code model | The Codex replacement (D1); not our contribution to reinvent |
| Prompt assembly | `pipeline/prompt.py` + `prompts/api.prompt` | **Port** official `chatapi.prompt` | The API spec IS the interface; must match to get comparable programs |
| Program compile/exec `φ` | `pipeline/executor.py` | **Reimplement** (sandboxed `exec`) | Core of the method; needs careful, safe control |
| `ImagePatch` / `VideoSegment` | `pipeline/image_patch.py`, `pipeline/video_segment.py` | **Port + reimplement** | The API surface programs call; port official semantics exactly |
| `find`, `exists` | `models/glip.py` | **Reuse** GLIP (official fork) | Detector; paper's choice, not our contribution. **Build from their fork** (OQ-8) |
| `verify_property`, `best_*_match` | `models/xvlm.py` | **Reuse** X-VLM checkpoint | Image-text matcher; paper's choice |
| `simple_query` | `models/blip2.py` | **Reuse** HF BLIP-2 `blip2-flan-t5-xxl` | Standard VQA backbone; HF-hosted |
| `compute_depth` | `models/depth.py` | **Reuse** MiDaS (torch.hub / HF) | Monocular depth; paper's choice |
| `llm_query`, `select_answer` | `models/text_llm.py` | **Reuse** open-weights instruct LLM | GPT-3 replacement (D2) |
| Model process bus | `pipeline/vision_bus.py` | **Reimplement** (simplified) | Official producer-consumer bus; we start single-process, add batching later |
| Datasets | `data/{refcoco,gqa,okvqa,nextqa}.py` | **Port** official dataset+metric code | Metric fidelity is critical; port `accuracy`/`post_process` verbatim |
| Metrics/eval | `eval/metrics.py` | **Port** official evaluators | Same reason; exact-match & VQA soft-acc & IoU |
| Figures | `eval/make_figures.py` | **Reimplement** | Publication figures from `results/` |
| Batch/smoke runners | `pipeline/run_batch.py`, `pipeline/run_smoke.py` | **Reimplement** | Orchestration + run-stamping |

**Principle:** reuse every pretrained perception/LLM component (none is the paper's
contribution); reimplement only the *program-synthesis-and-execution* core and the
orchestration/reproducibility scaffolding.

---

## 2. Finalized model choices (resolves OQ-5, OQ-6)

| Role | Choice | Fallback | Notes |
|------|--------|----------|-------|
| Program generator `π` (D1) | **Qwen2.5-Coder-7B-Instruct** | CodeLlama-7B-Instruct (matches official repo's `codellama` backend) | Strong Python synthesis; 7B fits alongside vision models. Try 14B/32B if memory allows — recorded as an ablation |
| Knowledge/MC LLM (D2) | **Llama-3.1-8B-Instruct** | Qwen2.5-7B-Instruct | For `llm_query` (OK-VQA) and `select_answer` (NExT-QA) |

Both run locally via `transformers`, temperature 0 (greedy) to mirror the paper's
deterministic Codex/GPT-3 settings. All swappable via `configs/` — no hardcoding.
An ablation over code-LLM size is planned to quantify D1's effect (the biggest
source of expected gap vs. the paper).

---

## 3. Pinned versions (version drift silently changes VLM results)

| Package | Pin | Why |
|---------|-----|-----|
| python | 3.10 | official repo baseline |
| torch / torchvision | 2.1.* / 0.16.* | GLIP fork's CUDA kernels target modern torch |
| pytorch-cuda | 12.1 | cluster CUDA module |
| transformers | 4.36.2 | BLIP-2 + Qwen/Llama support; avoid tokenizer regressions |
| accelerate | 0.25.0 | device_map for large models |
| bitsandbytes | 0.41.3 | BLIP-2 8-bit |
| omegaconf | 2.3.0 | config merge parity with official repo |
| timm / einops | 0.9.12 / 0.7.0 | GLIP / X-VLM deps |

Authoritative pin is `requirements.lock.txt` (generated in Phase 4). GLIP is built
from the official viper fork, not PyPI (OQ-8).

---

## 4. Config schema

- `configs/default.yaml` — base: seed, paths, `codegen`, `load_models`, thresholds,
  per-module settings, `dataset`, logging. (Already scaffolded.)
- `configs/experiment/repro_tableN_*.yaml` — overrides per target: dataset name/split,
  which models to load, batch size. (Already scaffolded, one per table.)
- Merge order: `default.yaml` ← experiment file ← CLI overrides (OmegaConf dotlist).
- **Rule:** every threshold/hyperparameter/model name lives in config. A magic number
  in `src/` is a bug caught in review.

---

## 5. Milestones (each ends runnable + testable)

1. **M0 — smoke path (CPU/toy).** `run_smoke` generates a program for a trivial query
   with a stub/tiny code-LLM and executes it against a tiny synthetic image with
   *mock* modules (find returns a fixed box, etc.). Toy end-to-end test < 1 min, no
   GPU. Proves executor + ImagePatch + prompt plumbing. → tag `smoke-pass`.
2. **M1 — single-stage correctness (GPU).** Each module wired to its real model and
   unit-tested on one real example (GLIP finds an object; BLIP-2 answers; X-VLM
   verifies; MiDaS depth; code-LLM emits valid `execute_command`). Assert shapes/dtypes
   at every boundary.
3. **M2 — Table 1 (RefCOCO) on a subset, then full.** Grounding needs the fewest
   modules (GLIP + executor + code-LLM). Sanity-check mean-IoU trend on ~100 samples,
   then full testA. First real reproduction number. → `repro-table1-v1`.
4. **M3 — Table 2 (GQA).** Adds BLIP-2 + X-VLM. Subset trend → full test-dev balanced.
5. **M4 — Table 3 (OK-VQA).** Adds `llm_query` (open LLM) + VQA soft-accuracy evaluator.
6. **M5 — Table 4 (NExT-QA).** Adds VideoSegment + `select_answer`; heaviest I/O.
7. **M6 — consolidation.** ≥3 seeds where stochastic, figures, deviations finalized,
   README from clean clone. → `thesis-submission`.

Order = increasing module count and increasing risk, so failures surface cheaply.

---

## 6. Compute budget (rough, inference-only)

| Milestone | GPU need | Est. GPU-hours | Notes |
|-----------|----------|---------------:|-------|
| M0 | none (CPU) | ~0 | mocks |
| M1 | 1×(24–80GB) | 1–2 | model load + single examples |
| M2 RefCOCO testA | 1×(40–80GB) | 8–20 | ~5–6k refs testA; GLIP + code-LLM per sample |
| M3 GQA test-dev | 1×(40–80GB) | 20–40 | ~12k balanced; +BLIP-2 XXL resident |
| M4 OK-VQA | 1×(40–80GB) | 10–20 | ~5k val; +text LLM |
| M5 NExT-QA | 1×(40–80GB) | 30–60 | video decode dominates |
| M6 (×3 seeds subsets) | 1×(40–80GB) | 20–40 | seeds on subsets where full is infeasible |

BLIP-2 flan-t5-xxl (8-bit ~20GB) + GLIP-L + a 7B code-LLM won't co-fit on 24GB;
plan for a **single 40–80GB GPU** (A100/H100), or use the multiprocessing path to
place models on separate GPUs. If only 24GB is available: fall back to
`blip2-flan-t5-xl` and a quantized code-LLM — recorded as deviations.

---

## 7. Risk register

| # | Risk | Likelihood | Mitigation / fallback |
|---|------|-----------|-----------------------|
| R1 | **GLIP fork won't build** (custom CUDA kernels vs cluster CUDA) | High | Build early on login node in M1; fallback to OWL-ViT/GroundingDINO for `find` (deviation) |
| R2 | **Open code-LLM emits invalid/degenerate programs** → big accuracy gap | High | Robust executor with retry + `fixed_code` fallback; ablate 7B/14B/32B; document as D1 |
| R3 | **Memory: models don't co-fit** | Med | 40–80GB GPU; else multiprocessing across GPUs, or smaller BLIP-2 variant |
| R4 | **Metric mismatch** (answer normalization, IoU convention) | Med | Port official `accuracy`/`post_process` verbatim; unit-test against known cases |
| R5 | **Offline weights missing on compute node** | Med | `verify_offline.py` on login node before every submit; check actual weight files |
| R6 | **Dataset access/licence** (NExT-QA videos, GQA size, disk quota) | Med | Download early; track quota in `cluster_notes.md`; subset if needed |

---

## 8. Definition of done for the build phases

Smoke reproducible from clean clone; `make test` green; every table row in
`results/reproduction_table.md` carries a non-dirty git SHA + run dir + config;
every deviation in code comment + `deviations.md` + README; ≥3 seeds where feasible;
figures regenerate via `make figures`.
