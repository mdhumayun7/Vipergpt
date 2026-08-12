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

### Confound resolved: the prompt, not the model

prompts/api.prompt is task-agnostic. Its in-context examples return strings
(`simple_query`, `best_text_match`, `llm_query`); nothing states that grounding
must return an ImagePatch. It also instructs explicitly:

  "Use base Python (comparison, sorting) for basic logical operations,
   left/right/up/down, math, etc."

So the near-absence of compute_depth (29) and distance (5) against find (547) is
the prompt working as designed, not the generator failing. The 98.8% task-invalid
return rate is a prompt-specification gap, not a Qwen limitation.

Next: prompts/api_grounding.prompt adds an explicit return-type contract plus two
grounding examples (one 2D, one depth-based). Measuring the before/after delta on
the same 500 queries. Original preserved as prompts/api.prompt.orig.

### Confound resolved: the prompt, not the model

prompts/api.prompt is task-agnostic. Its in-context examples return strings
(`simple_query`, `best_text_match`, `llm_query`); nothing states that grounding
must return an ImagePatch. It also instructs explicitly:

  "Use base Python (comparison, sorting) for basic logical operations,
   left/right/up/down, math, etc."

So the near-absence of compute_depth (29) and distance (5) against find (547) is
the prompt working as designed, not the generator failing. The 98.8% task-invalid
return rate is a prompt-specification gap, not a Qwen limitation.

Next: prompts/api_grounding.prompt adds an explicit return-type contract plus two
grounding examples (one 2D, one depth-based). Measuring the before/after delta on
the same 500 queries. Original preserved as prompts/api.prompt.orig.

### Confound resolved: the prompt, not the model

prompts/api.prompt is task-agnostic. Its in-context examples return strings
(`simple_query`, `best_text_match`, `llm_query`); nothing states that grounding
must return an ImagePatch. It also instructs explicitly:

  "Use base Python (comparison, sorting) for basic logical operations,
   left/right/up/down, math, etc."

So the near-absence of compute_depth (29) and distance (5) against find (547) is
the prompt working as designed, not the generator failing. The 98.8% task-invalid
return rate is a prompt-specification gap, not a Qwen limitation.

Next: prompts/api_grounding.prompt adds an explicit return-type contract plus two
grounding examples (one 2D, one depth-based). Measuring the before/after delta on
the same 500 queries. Original preserved as prompts/api.prompt.orig.

## 2026-08-11 — Prompt specification gap: quantified and fixed

Jobs 29187 (original prompt) and 29193 (grounding contract). Same 500 queries,
RefCOCO/testA and RefCOCO+/testA, Qwen2.5-Coder-7B-Instruct, greedy decoding.

| Metric                              | api.prompt | api_grounding.prompt |
|-------------------------------------|-----------:|---------------------:|
| Task-invalid return, RefCOCO        |      98.8% |                 0.2% |
| Task-invalid return, RefCOCO+       |      98.2% |                 0.0% |
| Spatial constraint dropped, RefCOCO |      17.3% |                 1.7% |
| Spatial constraint dropped, RefCOCO+|      31.9% |                 2.1% |
| Parse rate, RefCOCO                 |      99.8% |                96.6% |
| Parse rate, RefCOCO+                |      99.6% |                92.6% |
| Mean API calls, RefCOCO             |       2.45 |                 1.66 |

FINDING: the released ViperGPT prompt is task-agnostic. It never states that a
grounding program must return an ImagePatch, and its in-context examples all
return strings. With an open-weights generator this makes ~99% of programs
task-invalid — they parse, they run, and they return a sentence where a box is
required. Adding an explicit return-type contract and two grounding examples
reduces this to 0.2%.

COST: parse rate falls 99.8 -> 96.6 (RefCOCO) and 99.6 -> 92.6 (RefCOCO+). The
stricter contract makes a small fraction of generations syntactically invalid.
Reported, not hidden.

Consistency: a 30-query pilot gave 0.0%, the 500-query run gave 0.2%.

This is a reproducibility result. Codex is retired, so anyone reproducing ViperGPT
today must substitute an open model, and will hit this wall unannounced — the
paper does not mention the contract because Codex apparently inferred it.

## 2026-08-11 — Milestone 2: GLIP built and verified on H100

GLIP compiles and runs on sm_90. CUDA NMS verified independently (keep=[0,2] on a
hand-checked case). Inference ~0.15 s/image on H100 NVL.

Three fixes were needed beyond the CUDA header work (D5):

1. **D6 — NumPy 2.0 aliases.** 5 files used `np.float`, removed in NumPy 1.24.
   Construction failed in anchor_generator.py before any inference.
2. **D7 — BERT tokenizer downloaded at load time.** GLIPDemo fetches
   bert-base-uncased (440MB) on first construction; now cached, but SLURM jobs must
   set HF_HUB_OFFLINE=1 or they will hang on an offline compute node.
3. **The decisive one — `tensor_inputs`.** GLIPDemo defaults to `tensor_inputs=False`,
   which assumes HWC numpy input and derives the image size as
   `original_image.shape[:-1]`. We pass CHW tensors, so it read (3, H) and rescaled
   every box into a 3-pixel-wide image. Every IoU was *exactly* 0.000 — not low,
   zero — which is what pointed at a coordinate-space bug rather than a detection
   quality problem. Passing `tensor_inputs=True` selects `build_tensor_transforms()`
   and `shape[-2:]`, and fixes it.

Validation against RefCOCO ground truth (20 samples, testA, threshold 0.5,
GLIP prompted with the head noun only):

| Metric | Value |
|---|---|
| mean IoU (flipped, as returned) | 0.491 |
| mean IoU (unflipped control) | 0.364 |
| IoU >= 0.5 | 41.7% |
| no detection | 8/20 |

Flipped scoring beats unflipped, which confirms the bottom-left origin convention
rather than assuming it. Best individual cases: 0.951, 0.949, 0.856.

Caveats: this is raw GLIP top-box accuracy, NOT ViperGPT accuracy — no program is
executed and no spatial reasoning is applied, so it is not comparable to the paper's
72.0. The 40% no-detection rate needs investigation; the head-noun heuristic is crude
(`"player number 8"` -> `"8"`, which is discarded by the isalpha filter).

| no detection | 8/20 |

Threshold sweep (same 20 samples):

| Threshold | Scored | Mean IoU | IoU>=0.5 |
|---|---|---|---|
| 0.5 | 12/20 | 0.491 | 41.7% |
| 0.3 | 19/20 | 0.578 | 52.6% |
| 0.2 | 19/20 | 0.776 | 78.9% |

The default 0.5 is badly miscalibrated for this use: it loses 8 of 20 detections
outright. At 0.2, detections do not increase further (19 either way) but mean IoU
rises sharply, so the lower threshold is selecting better boxes rather than simply
returning more. Threshold to be fixed by config, recorded as a deviation, and not
tuned per-experiment.

Caveats: this is raw GLIP top-box accuracy, NOT ViperGPT accuracy. No program is
executed and no spatial reasoning is applied, so 78.9% is not comparable to the
paper's 72.0 on RefCOCO. The head-noun heuristic is also crude: "player number 8"
reduces to "8", which the isalpha filter then discards.

## 2026-08-12 — Milestone 2: end-to-end execution, first real accuracy

Same 50 programs from each Milestone 1 run, executed through GLIP(0.2) + CLIP(D9)
+ MiDaS. Non-patch returns score 0 rather than being excluded — a program that
returns a sentence has failed the task, and dropping it would flatter the baseline.

| Metric | Baseline prompt | + grounding contract |
|---|---:|---:|
| Accuracy IoU>=0.5 | 0.00% | 36.00% |
| Mean IoU | 0.0000 | 0.3508 |
| Returned a patch | 0/50 | 44/50 |
| Runtime errors | 45 | 6 |
| spatial acc (n=28) | 0.00% | 28.57% |
| non-spatial acc (n=22) | 0.00% | 45.45% |

The Milestone 1 static prediction — that ~99% of baseline programs are structurally
incapable of scoring — is confirmed by execution: zero of fifty returned a patch.

CAVEAT, to resolve before reporting: baseline failure is over-determined. Of 50
programs, 45 raised at execution (KeyError x28) and only 5 got as far as returning
a string. The KeyErrors are almost certainly `llm_query`, which the baseline prompt
invoked 265 times in the 500-program trace and which this bus does not register
(load_models.llm_qa=False). So the baseline scores 0% for two independent reasons:
wrong return type, and a module we chose not to load. These must be separated —
either enable llm_qa, or report the two failure modes distinctly. As it stands the
0% is correct but not cleanly attributable.

Grounding-run errors (6) are AttributeError x4, NameError, KeyError — to be
inspected individually.

Context for the 36%: the paper reports 72.0. Raw GLIP top-box accuracy on this data
is 78.9% (threshold 0.2), so perception caps us well below the paper before any
program runs. Known contributors to the gap: D1 (Codex->Qwen), D9 (X-VLM->CLIP),
D8 (threshold tuned on only 20 samples).

Notable: spatial queries score 28.57% against 45.45% for non-spatial. That gap is
the dissertation's actual target.

Grounding-run errors (6) are AttributeError x4, NameError, KeyError.

### Confound removed (D10)

The first run's baseline was over-determined: 28 of 45 errors were KeyError from
simple_query (10) and llm_query (18), modules this bus does not load. That made the
0% partly OUR configuration's fault, not the programs'. Registering neutral
empty-string fallbacks for those calls lets every program run to completion.

| Metric | Baseline v2 | Grounding v2 |
|---|---:|---:|
| Accuracy IoU>=0.5 | 0.00% | 36.00% |
| Mean IoU | 0.0000 | 0.3569 |
| Returned a patch | 0/50 | 45/50 |
| Returned a string | 33/50 | 0/50 |
| Errors | 17 | 5 |

33 baseline programs now execute cleanly end to end and still score exactly zero,
because they return a sentence where a box is required. The failure is now
attributable to the return type alone. The fallbacks change grounding by almost
nothing (36.00% either way, mean IoU 0.3508 -> 0.3569), which is itself evidence:
programs written under the grounding contract do not need the text-answering calls.

Remaining errors are genuine program bugs, and are a finding in their own right —
Qwen invents API surface that does not exist: `crop_area`, `.area`, `.text`,
and the builtin `next` (deliberately absent from the executor's safe builtins).
Two `.text` cases were queries requiring OCR ("number 8", "2"), a capability the
ViperGPT API does not expose at all.