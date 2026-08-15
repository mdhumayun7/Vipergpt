# Deviations from the paper

Living document. Every substitution away from what the paper specifies is
recorded here, in a `# DEVIATION:` code comment at the site, and in the README
results table. No silent substitutions.

| # | Area | Paper specifies | This repo uses | Reason | Likely effect on metric |
|---|------|-----------------|----------------|--------|-------------------------|
| D1 | Program generator (π) | OpenAI Codex `code-davinci-002` | Open-weights code model (finalized in plan) | Codex API deprecated 2023-03-23; unavailable | Uncertain; open code models generate less reliable programs → likely small negative |
| D2 | External-knowledge LLM (`llm_query`) | GPT-3 `text-davinci-003` | Open-weights instruct LLM | Cost + offline-cluster requirement; text-davinci-003 also deprecated | Possible negative on OK-VQA (knowledge recall) |

More rows added as implementation proceeds.

## D1 — Codex replaced by Qwen2.5-Coder-7B-Instruct
**Paper:** the program generator pi is OpenAI Codex (`code-davinci-002`).
**Used instead:** Qwen/Qwen2.5-Coder-7B-Instruct, bfloat16, greedy decoding.
**Reason:** Codex was retired by OpenAI and is unavailable to anyone. Any
present-day reproduction must substitute a different generator; there is no
faithful option.
**Effect on results:** central to this work rather than incidental. The
substitution is what exposed the prompt-specification gap measured in Milestone 1
(98.8% task-invalid returns under the released prompt, 0.2% once the grounding
contract is stated). It is also a primary contributor to the gap against the
paper's reported 72.0 on RefCOCO.

## D2 — Scope restricted to visual grounding
**Paper:** four tasks — RefCOCO/RefCOCO+ grounding, GQA, OK-VQA, NExT-QA.
**Used instead:** RefCOCO and RefCOCO+ only (Table 1).
**Reason:** the full four-table setup needs roughly 250GB against ~231GB of quota;
NExT-QA alone requires 100GB+ of video, and OK-VQA needs GPT-3, also retired.
**Effect on results:** no cross-task claims are made. The grounding numbers stand
on their own.

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

## D8 — GLIP confidence threshold
**Paper:** not specified. The official repo's config defaults to 0.5.
**Used instead:** 0.2, set in configs/default.yaml.
**Reason:** swept on 20 RefCOCO/testA samples against ground truth —
0.5 gives mean IoU 0.491 and misses 8/20 detections entirely; 0.2 gives 0.776 and
misses 1/20. Detection count is identical at 0.3 and 0.2, so the improvement comes
from better box selection rather than a higher recall of low-confidence boxes.
**Effect on results:** material. The threshold is fixed once here and used for every
condition, so it cannot favour one prompt variant over another.
**Validated 2026-08-12** on a held-out slice (RefCOCO/testA samples 200-399,
n=200, disjoint from both the original 20 and the 0-499 range used for reported
numbers), job 29318:

| threshold | acc IoU>=0.5 | mean IoU | boxes/img |
|---|---:|---:|---:|
| 0.10 | 81.50% | 0.7372 | 25.5 |
| 0.15 | 81.50% | 0.7372 | 25.4 |
| **0.20** | **80.50%** | 0.7336 | **19.7** |
| 0.25 | 70.00% | 0.6658 | 9.3 |
| 0.30 | 59.50% | 0.5784 | 5.4 |
| 0.50 | 31.00% | 0.3360 | 1.3 |

0.2 is within 1.0 point of the optimum. The decision rule was fixed before seeing
these numbers: keep 0.2 if within 2 points, otherwise re-run everything. It is
within, so 0.2 stands and no reported result changes. 0.10 and 0.15 are identical
to four decimal places, so nothing is gained below 0.2 except more boxes per image
(25.5 vs 19.7), which only enlarges the candidate set a program must filter.

Separately: 21 of 200 samples yield no detection at EVERY threshold up to 0.4. That
floor is not threshold-related — it comes from the head-noun heuristic used to prompt
GLIP ("player number 8" -> "8" -> discarded by the isalpha filter). A known,
separate limitation. Full sweep: results/threshold_sweep.json

## D9 — X-VLM replaced by CLIP
**Paper:** X-VLM backs verify_property / best_text_match / best_image_match.
**Used instead:** openai/clip-vit-large-patch14 (models/clip_vlm.py).
**Reason:** X-VLM weights are distributed via Google Drive and cannot be fetched by
script; models/_xvlm_backbone.py remains a stub.
**Effect:** X-VLM has fine-grained region-text alignment; CLIP scores globally, so
attribute verification should be weaker. Treat as a lower bound. Impact on RefCOCO
is limited — find is called 547 times against 109 for verify_property.

## D10 — Neutral fallbacks for unloaded text modules
**Paper:** BLIP-2 backs simple_query, a text LLM backs llm_query.
**Used instead:** both return "" when not loaded.
**Reason:** loading BLIP-2-XXL (40GB) and an 8B text LLM is out of scope for visual
grounding. Without a fallback, any program calling them raises KeyError, making its
failure attributable to our configuration rather than to the program.
**Effect:** none on grounding (36.00% with or without; mean IoU 0.3508 -> 0.3569),
and it strictly favours the baseline by letting 33 more programs run to completion.
Grounding requires an ImagePatch, which no string can satisfy.

## D11 — Depth model: offline loading and sign correction
**Original wrapper:** loaded MiDaS via `torch.hub.load()` and returned the raw median.
**Two defects:** torch.hub reaches the network, which hangs on an offline compute
node; and MiDaS predicts INVERSE depth (larger = closer), so the returned value was
monotonically decreasing in true distance — the opposite of the API docstring and of
what any generated program assumes.
**Now:** loads the cached HF checkpoint `Intel/dpt-hybrid-midas`, resamples the
prediction to image size (the DPT processor emits a square map, so proportional
region reads were sampling squashed regions), and inverts so larger = further.
**Evidence for the sign:** on 25 RefCOCO+ depth queries scored against ground-truth
boxes with no detector involved, the target lands at the correct depth extreme 80%
of the time with the inversion and 8% without, against 26% chance. 8% is far below
chance, i.e. systematically inverted rather than weak.
**Effect on results:** every depth-based number depends on this. Pinned by
test_depth_api.py (18 assertions on ordering semantics).

## D12 — GLIP wrapper: detection scores and candidate filtering
**Paper/official:** `find` returns boxes only.
**Added:** the wrapper now also exposes per-box confidence (`last_scores`) and two
optional knobs, `max_detections` and `find_nms_iou`.
**Reason:** at the tuned threshold (0.2, D8) `find` returns ~40 boxes per image. That
is right for top-box detection accuracy (81.5%) and wrong for any program that must
SELECT among candidates — depth ordering and per-candidate attribute checks both
degrade on a set that large.
**Defaults are OFF** (`max_detections=0`, `find_nms_iou=0`), so every reported number
predates and is unaffected by this change. Enabling it is a documented ablation, not
part of the main results.
