# Research log

Dated, append-only. Records what was tried, what happened, and why — including
failures and discarded approaches. This is the source for the thesis's
"challenges faced" section.

## 2026-08-10
- Project kickoff. Target: reproduce ViperGPT Tables 1–4 (RefCOCO/RefCOCO+,
  GQA, OK-VQA, NExT-QA) on a university SLURM GPU cluster.
- Read the full paper (arXiv:2303.08128v1, 14 Mar 2023) and the official repo
  (github.com/cvlab-columbia/viper).
- Key early finding: Codex (`code-davinci-002`) is deprecated. The official
  repo already provides GPT-3.5/GPT-4 and a **CodeLlama** backend as
  alternatives, confirming the open-weights route is viable. Decision (with
  supervisor sign-off pending): use an open-weights code model as π.
- Phase 0 scaffold created.
- Resolved OQ-1 from official `datasets/refcoco.py`: reported RefCOCO number is
  **mean IoU** (secondary [email protected]); boxes use a bottom-left
  `(left,lower,right,upper)` convention — flagged as a bug trap.
- Phases 2 (spec) and 3 (plan) written and signed off. Decisions locked:
  Qwen2.5-Coder-7B as π; build all four tables then run.
- Phase 5 implemented: config/prompt/executor/ImagePatch/VideoSegment/vision-bus,
  model wrappers (GLIP/X-VLM/BLIP-2/MiDaS/text-LLM), 4 dataset loaders, evaluators,
  run_batch/run_smoke, figures, and tests (metrics/executor/image_patch/smoke).
- **Environment failure:** the dev shell never booted this session, so tests are
  unrun and commits are queued. `git init` + tags + `make test` are the first
  actions to take once a shell is available. Actual reproduction requires the GPU
  cluster (weights + datasets).
- Coordinate-convention note (`ImagePatch.__init__` uses `image[:, lower:upper, ...]`
  on a top-left-origin tensor): needs empirical verification in milestone M1 against
  a real GLIP detection before trusting Table 1 numbers.

## 2026-08-10 — Environment bring-up and Milestone 1

**Cluster:** svnithpc, SLURM, 2 nodes, 1x H100 NVL (95GB) each, fractional via
`--gres=shard:N` out of 94. No `/scratch`; `/data` (18TB) is root-owned and not
writable by this account. All work therefore lives under `/home/mazaveri`
(quota 500G, 269G used; shared filesystem was 93% full throughout).

**What worked, in order:**
- Env `vipergpt` already existed, so `conda env create` was a no-op. Package
  installed with `python -m pip install -e .`
- `make smoke` and `make test` pass — 20 tests, mock module path OK.
- RefCOCO/RefCOCO+ annotations downloaded, validated (19994/19992 images).
- Qwen2.5-Coder-7B-Instruct cached, 4 shards, 15.2 GB.
- Job 29187: full preflight PASS, generation running on cuda:0 at ~8 queries/25s.

**Problems hit and fixes:**

1. *pip/python mismatch.* `conda activate vipergpt` put python 3.10 on PATH but
   `pip` still resolved to base anaconda 3.11, so `pip install -e .` installed
   into `~/.local/lib/python3.11` while `make smoke` ran the env's python →
   `ModuleNotFoundError: vipergpt_repro`. Fix: always `python -m pip`, never bare
   `pip`. Added a `sys.prefix` assertion to the preflight.

2. *Version drift.* Because the env pre-existed, `environment.yml` pins were never
   applied. Installed stack is torch 2.13.0+cu130 / transformers 5.15.0, not the
   pinned 2.1 / 4.36.2. Left as-is for Milestone 1 (mock + text-only path is
   unaffected). FLAGGED AS A RISK for Milestone 2: the GLIP fork's custom CUDA
   kernels predate sm_90 and are unlikely to build against CUDA 13.

3. *RefCOCO host is dead.* `bvisionweb1.cs.unc.edu` no longer resolves from any
   network tested. Switched to Internet Archive snapshots (see D3). Download ran
   at ~111 KB/s, ~15 min for both zips. HuggingFace by contrast ran at ~4 MB/s.

4. *`hf download --exclude` crash.* `--exclude "*.pth" "original/*"` was parsed as
   positional filenames; the CLI printed help and exited 1, so the first Qwen
   download silently never started (cache stayed at 6.8 MB). Removed the flag.

5. *`conda activate` fails silently in a batch shell.* Job 29186 reported
   `Python 3.11.7` from base anaconda inside the SLURM job even after
   `conda activate vipergpt`, which then loaded the broken `~/.local` torch
   (`OSError: libtorch_global_deps.so`). Preflight caught it and aborted before
   wasting the allocation. Fix: prepend the env's bin to PATH explicitly and set
   `PYTHONNOUSERSITE=1`.

6. *SLURM log dir.* `outputs/slurm/` must exist before `sbatch`, or the job is
   discarded with no `.out` and no `.err` at all.

**Open issue — split mismatch (NOT yet resolved):**
Phase 1 reports `refcoco: splits ['test','train','val']` but
`refcoco+: splits ['testA','testB','train','val']`. The paper evaluates both on
testA. The RefCOCO zip appears to have supplied the google split rather than unc.
Comparing RefCOCO/test against RefCOCO+/testA is not a like-for-like control and
must be fixed before any number is cited.

**Design decision.** Scoped to RefCOCO/RefCOCO+ (Table 1) only. GQA, OK-VQA and
NExT-QA are out of scope — NExT-QA alone needs 100GB+ of video, and the full
four-table setup needs ~250GB against ~231GB of quota. Milestone 1 (program
generation analysis) was deliberately sequenced before Milestone 2 (execution)
because it needs only the code generator, and therefore yields a citable result
even if the GLIP build fails.

## 2026-08-11 — Milestone 1 result: initial hypothesis NOT supported

Job 29187, 3818s, all stages OK. 500 queries each on RefCOCO/testA and
RefCOCO+/testA, Qwen2.5-Coder-7B-Instruct, greedy.

| Dataset  | Subset      | N   | Parse% | Collapse% | Mean API |
|----------|-------------|-----|--------|-----------|----------|
| RefCOCO  | all         | 500 | 99.8   | 0.6       | 2.45     |
| RefCOCO  | spatial     | 289 | 100.0  | 0.3       | 2.35     |
| RefCOCO  | non-spatial | 211 | 99.5   | 1.0       | 2.60     |
| RefCOCO+ | all         | 500 | 99.6   | 0.2       | 2.89     |
| RefCOCO+ | spatial     | 47  | 97.9   | 0.0       | 3.33     |
| RefCOCO+ | non-spatial | 453 | 99.8   | 0.2       | 2.84     |

HYPOTHESIS REJECTED. We predicted that spatial-relational queries would cause the
generator to collapse the whole query into a single opaque `simple_query` call.
Collapse rate is ~0% everywhere. The reason: the `simple_query` shortcut for
relational terms lives in the paper's GQA prompt, not the grounding prompt, and
visual grounding must return an ImagePatch, which `simple_query` cannot produce.
The measurement was well-formed but aimed at the wrong task.

Cost of finding this out: one 63-minute job, before any perception model was built.

NEW QUESTION, from the same data: spatial queries in RefCOCO use FEWER API calls
(2.35) than non-spatial ones (2.60), and RefCOCO+ overall uses more (2.89) than
RefCOCO (2.45). If spatial relations are being resolved by 2D pixel arithmetic on
`find()` output rather than by the geometric API, then `compute_depth`/`distance`
should be near-absent from the call-frequency table. Checking next.

NOTE: Phase 1 of run_all.sh reports RefCOCO splits as ['test','train','val'],
which is misleading — that checker reads refs(google).p, while the analysis script
is patched to prefer refs(unc).p and did use testA. Phase 1's checker should be
patched to match.

## 2026-08-11 — Milestone 1 result: initial hypothesis NOT supported

Job 29187, 3818s, all stages OK. 500 queries each on RefCOCO/testA and
RefCOCO+/testA, Qwen2.5-Coder-7B-Instruct, greedy.

| Dataset  | Subset      | N   | Parse% | Collapse% | Mean API |
|----------|-------------|-----|--------|-----------|----------|
| RefCOCO  | all         | 500 | 99.8   | 0.6       | 2.45     |
| RefCOCO  | spatial     | 289 | 100.0  | 0.3       | 2.35     |
| RefCOCO  | non-spatial | 211 | 99.5   | 1.0       | 2.60     |
| RefCOCO+ | all         | 500 | 99.6   | 0.2       | 2.89     |
| RefCOCO+ | spatial     | 47  | 97.9   | 0.0       | 3.33     |
| RefCOCO+ | non-spatial | 453 | 99.8   | 0.2       | 2.84     |

HYPOTHESIS REJECTED. We predicted that spatial-relational queries would cause the
generator to collapse the whole query into a single opaque `simple_query` call.
Collapse rate is ~0% everywhere. The reason: the `simple_query` shortcut for
relational terms lives in the paper's GQA prompt, not the grounding prompt, and
visual grounding must return an ImagePatch, which `simple_query` cannot produce.
The measurement was well-formed but aimed at the wrong task.

Cost of finding this out: one 63-minute job, before any perception model was built.

NEW QUESTION, from the same data: spatial queries in RefCOCO use FEWER API calls
(2.35) than non-spatial ones (2.60), and RefCOCO+ overall uses more (2.89) than
RefCOCO (2.45). If spatial relations are being resolved by 2D pixel arithmetic on
`find()` output rather than by the geometric API, then `compute_depth`/`distance`
should be near-absent from the call-frequency table. Checking next.

NOTE: Phase 1 of run_all.sh reports RefCOCO splits as ['test','train','val'],
which is misleading — that checker reads refs(google).p, while the analysis script
is patched to prefer refs(unc).p and did use testA. Phase 1's checker should be
patched to match.

### Follow-up analysis (same data, no GPU)

| Run | Returns a string (task-invalid) | Spatial query with no positional op |
|---|---:|---:|
| RefCOCO/testA  | 98.8% | 17.3% |
| RefCOCO+/testA | 98.2% | 31.9% |

Visual grounding requires an ImagePatch (a box). Essentially every generated
program returns a natural-language string instead. Programs parse at ~100% and are
task-invalid at ~99%. Predicted RefCOCO IoU accuracy under execution: near zero.

Call frequency confirms the geometric API is barely touched: 547 `find` against
29 `compute_depth` and 5 `distance`. Where spatial relations ARE handled, it is via
2D pixel arithmetic on patch attributes (`max(patches, key=lambda p: p.right)`),
never via geometry. This explains the earlier anomaly that spatial queries use
fewer API calls than non-spatial ones — the spatial constraint is often dropped
outright (17.3% / 31.9% of spatial queries contain no positional operation at all).

`llm_query` appears 265 times — a text-only model being asked image questions
("Who is in the bottom left corner of the image?"), which cannot work.

CONFOUND, to resolve before claiming anything: is this Qwen's failure or
ViperGPT's? Codex may have handled the API contract correctly. Next step is to
check whether prompts/api.prompt carries grounding-specific examples showing that
execute_command must return a patch. If adding them moves the 98.8% materially,
the finding is about prompt/model transfer; if not, it is a structural limitation
of open-weights substitution — which is itself the reproducibility result, since
Codex is no longer available to anyone.
