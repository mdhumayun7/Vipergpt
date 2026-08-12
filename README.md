# ViperGPT — Reproduction

An independent, thesis-grade reproduction of **ViperGPT: Visual Inference via Python
Execution for Reasoning** (Surís, Menon, Vondrick, ICCV 2023 —
[arXiv:2303.08128](https://arxiv.org/abs/2303.08128),
[official code](https://github.com/cvlab-columbia/viper)).

**Targets:** Tables 1–4 — RefCOCO/RefCOCO+ (grounding), GQA (compositional VQA),
OK-VQA (knowledge VQA), NExT-QA (video reasoning).

> **Build status.** The full pipeline, datasets, metrics, and a CPU/mock smoke path
> are implemented. The reproduction *numbers* (Phase 6) require the university GPU
> cluster with the pretrained weights and datasets downloaded — see
> [Running on the cluster](#running-on-the-cluster). The unit/smoke tests were
> written but **not yet executed in this environment** (the dev shell was
> unavailable); run `make test` from a clean clone to verify — this is the first
> checklist item before citing any result.

## Method in one paragraph
A code-generation LLM `π` turns a query into a Python program calling a fixed visual
API (`find`, `simple_query`, `verify_property`, `compute_depth`, `llm_query`, …). A
restricted interpreter `φ` runs that program, dispatching each API call to a
pretrained module (GLIP, X-VLM, BLIP-2, MiDaS, a text LLM). The framework is
training-free. See [`docs/paper_spec.md`](docs/paper_spec.md).

## Key deviations (mandatory — see [`docs/deviations.md`](docs/deviations.md))
- **D1:** program generator Codex `code-davinci-002` (deprecated 2023-03-23) →
  **Qwen2.5-Coder-7B-Instruct** (open weights). Fallback: CodeLlama-7B.
- **D2:** `llm_query`/`select_answer` GPT-3 `text-davinci-003` (deprecated) →
  **Llama-3.1-8B-Instruct** (open weights).

## Quickstart (local, no GPU) — verifies the code path
```bash
conda env create -f environment.yml && conda activate vipergpt
pip install -e .
make test        # unit + toy end-to-end tests (mock modules, CPU)
make smoke       # runs the toy pipeline end to end
```

## Running on the cluster
1. **[LOGIN NODE]** download weights + data (no internet on compute nodes):
   ```bash
   PRETRAINED_MODEL_PATH=/scratch/$USER/vipergpt/pretrained_models bash scripts/download_models.sh
   DATA_PATH=/scratch/$USER/vipergpt/data bash scripts/download_data.sh
   ```
   Build the GLIP fork per the official repo (custom CUDA kernels; see
   `docs/cluster_notes.md`).
2. **[LOGIN NODE]** verify offline load before submitting:
   ```bash
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_offline.py
   ```
3. **Submit** a reproduction (one per table):
   ```bash
   sbatch --export=ALL,EXP=repro_table1_refcoco scripts/slurm/eval.sbatch
   sbatch --export=ALL,EXP=repro_table2_gqa      scripts/slurm/eval.sbatch
   sbatch --export=ALL,EXP=repro_table3_okvqa    scripts/slurm/eval.sbatch
   sbatch --export=ALL,EXP=repro_table4_nextqa   scripts/slurm/eval.sbatch
   ```
   Each run writes a stamped directory under `outputs/runs/` (git SHA, config, seed,
   hardware) with `records.jsonl` and `metrics.json`.
4. **Figures:** `make figures` → `results/figures/reported_vs_reproduced.pdf`.

## Repository map
| Path | Role |
|------|------|
| `src/vipergpt_repro/pipeline/config.py` | OmegaConf load/merge |
| `src/vipergpt_repro/pipeline/prompt.py` + `prompts/api.prompt` | API-spec prompt assembly |
| `src/vipergpt_repro/models/codegen.py` | program generator π (Qwen2.5-Coder) |
| `src/vipergpt_repro/pipeline/executor.py` | restricted-namespace program execution φ |
| `src/vipergpt_repro/pipeline/image_patch.py` | `ImagePatch` + free functions (API surface) |
| `src/vipergpt_repro/pipeline/video_segment.py` | `VideoSegment` (video API surface) |
| `src/vipergpt_repro/pipeline/vision_bus.py` | module dispatch (real + mock backends) |
| `src/vipergpt_repro/models/{glip,xvlm,blip2,depth,text_llm}.py` | pretrained module wrappers |
| `src/vipergpt_repro/data/{refcoco,gqa,okvqa,nextqa}.py` | dataset loaders (ported) |
| `src/vipergpt_repro/eval/metrics.py` | IoU / VQA-acc / soft-acc / MC-acc evaluators |
| `src/vipergpt_repro/pipeline/run_batch.py` | reproduction runner |
| `src/vipergpt_repro/pipeline/run_smoke.py` | CPU toy end-to-end |

## Results
See [`results/reproduction_table.md`](results/reproduction_table.md) (populated in
Phase 6). Every cited row must carry a non-dirty git SHA + run dir + config.

## Citation
See [`CITATION.cff`](CITATION.cff) and [`paper/references.bib`](paper/references.bib).
Licensed MIT (this reproduction); the original work belongs to its authors.


## Status — 11 August 2026

**Milestone 1 (complete, tag `m1-complete`).** Program-generation analysis on
RefCOCO/RefCOCO+ testA, 500 queries each, Qwen2.5-Coder-7B-Instruct.
The released ViperGPT prompt does not state that a grounding program must return an
`ImagePatch`; with an open-weights generator, 98.8% of programs return a string
instead. Adding an explicit return-type contract and two grounding exemplars reduces
this to 0.2%. Parse rate falls 99.8 -> 96.6 as a cost. See `docs/research_log.md`.

**Milestone 2 (in progress).** Perception stack built and verified:
- GLIP compiled for H100 / sm_90 (D5, D6); CUDA NMS verified independently.
- GLIP validated against RefCOCO ground truth: mean IoU 0.776, IoU>=0.5 78.9%
  at threshold 0.2 (D8). Coordinate convention confirmed empirically, not assumed.
- COCO train2014 (82,783 images), MiDaS, and the GLIP-L checkpoint are on disk.

Not yet done: X-VLM wrapper (`models/_xvlm_backbone.py` is still a stub — see D9
plan to substitute CLIP), the program executor, and end-to-end IoU accuracy.

### Environments

Two conda environments, deliberately separate and never active together:

| Env | Purpose | Key pins |
|---|---|---|
| `vipergpt` | code generation (Qwen) | torch 2.13, transformers 5.15 |
| `glip_env` | perception (GLIP, MiDaS) | torch 2.1.2+cu121, transformers 4.36.2, numpy<2 |

GLIP's fork requires the older stack; Qwen does not. Generation writes
`programs.jsonl`, execution reads it, so the two stages never need one interpreter.
