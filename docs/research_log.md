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
## 2026-08-12 — Milestone 2 full run (n=500), job 29290

Confirmed on GPU (node2, H100 NVL). An earlier 500-sample attempt ran on the LOGIN
node and produced RuntimeError x481 with a 7-second "grounding" run; those results
were discarded. The job script now asserts torch.cuda.is_available() and aborts
otherwise. Silent CPU fallback is the most dangerous failure mode seen so far — it
produces a complete, plausible, entirely fake results table.

RefCOCO/testA, 500 programs per condition, GLIP(0.2) + CLIP(D9) + MiDaS, D10 fallbacks.

| Metric | Baseline prompt | + grounding contract |
|---|---:|---:|
| Accuracy IoU>=0.5 | 0.00% | 39.20% |
| Mean IoU | 0.0000 | 0.3856 |
| Returned a patch | 0/500 | 443/500 |
| Returned a string | 336/500 | 1/500 |
| Errors | 163 | 56 |
| spatial acc (n=289) | 0.00% | 35.29% |
| non-spatial acc (n=211) | 0.00% | 44.55% |

Runtimes: baseline 230s, grounding 500s. The grounding run is slower because its
programs actually call the perception stack instead of returning early.

SECOND EFFECT OF THE CONTRACT, not anticipated: under the baseline prompt, 136 of
500 generations (27.2%) did not define `execute_command` at all — the model emitted
prose or malformed markdown that the extractor could not recover. Under the
grounding prompt this drops to zero. The contract disciplines output FORMAT as well
as return type; these are separable failures and should be reported separately.

Baseline scores exactly 0/500 with 336 programs running to completion and returning
a string. No configuration excuse remains: the failure is the return type.

Spatial queries score 35.29% against 44.55% for non-spatial — a 9.3-point gap on
identical infrastructure. This is the quantity the dissertation contribution targets.

Context: the paper reports 72.0 on RefCOCO. Raw GLIP top-box accuracy on this data
is 78.9% (n=20, threshold 0.2), so perception alone caps us near 79% and program
selection loses ~40 points from there. Known contributors: D1 (Codex->Qwen),
D9 (X-VLM->CLIP), D8 (threshold tuned on only 20 samples).

## 2026-08-12 — RefCOCO+ control (job 29292, n=500)

| Metric | Baseline | + grounding contract |
|---|---:|---:|
| Accuracy IoU>=0.5 | 0.00% | 31.80% |
| Mean IoU | 0.0000 | 0.3245 |
| Returned a patch | 0/500 | 386/500 |
| No execute_command defined | 214/500 | 0 |
| spatial acc (n=47) | 0.00% | 6.38% |
| non-spatial acc (n=453) | 0.00% | 34.44% |

PREDICTION FALSIFIED. We expected the spatial/non-spatial gap to shrink or vanish
on RefCOCO+, since that dataset forbids spatial relations. It widened instead:

| Dataset | Spatial | Non-spatial | Gap |
|---|---:|---:|---:|
| RefCOCO  | 35.29% (n=289) | 44.55% (n=211) | 9.3 pts |
| RefCOCO+ | 6.38% (n=47)   | 34.44% (n=453) | 28.1 pts |

Reading: because RefCOCO+ forbids spatial language, the 47 queries our lexicon
flags there are not ordinary spatial queries — they are annotation-guideline
violations or words used non-spatially. They are edge cases, and the system fails
on them almost completely. This does not contradict the spatial hypothesis; it says
the RefCOCO+ spatial subset is not the clean control we assumed. n=47 means 6.38%
is 3 of 47, so the interval is wide — the direction is informative, the value is not.

Overall RefCOCO+ accuracy is also lower than RefCOCO (31.80% vs 39.20%). Expected:
RefCOCO+ expressions are attribute-based, so they lean on verify_property, which is
CLIP here rather than X-VLM (D9). This is the deviation most likely to be costing us.

The format effect is stronger here: 214/500 baseline generations (42.8%) never
defined execute_command, against 136/500 (27.2%) on RefCOCO. Zero under the
grounding contract on both.

### Manual audit of the RefCOCO+ spatial subset (n=47)

The "edge case / annotation violation" reading was wrong. The 47 flagged queries are
genuinely spatial — they are simply a DIFFERENT KIND of spatial from RefCOCO's:

  RefCOCO   : 2D image-plane      "man on right", "person bottom left", "left kid"
  RefCOCO+  : depth & relational  "man closest to us", "farthest man", "surfer
                                   closest", "umpire behind catcher", "nearest guy",
                                   "woman between man and dark hair woman"

This is why RefCOCO+ forbidding spatial language did not remove spatial queries: it
removed *viewer-frame directional* language, and what remains is depth-ordering and
inter-object relations.

The distinction is exactly the one the dissertation targets:

| Spatial type | Resolvable by | Accuracy |
|---|---|---:|
| 2D image-plane (RefCOCO) | base Python on patch attributes | 35.29% |
| Depth / relational (RefCOCO+) | requires compute_depth or 3D geometry | 6.38% |

2D relations are handled by sorting patch coordinates
(`sort(key=lambda p: p.horizontal_center)`), which the released prompt explicitly
endorses. Depth relations cannot be. Milestone 1 showed compute_depth is invoked 29
times against 547 for find, so in practice depth is essentially never used — and the
queries that need it score 6.38%.

Supporting detail: all three hits in this subset are depth queries scoring very high
(0.97 "umpire behind catcher", 0.96 "girl closest to hydrant", 0.92 "farthest man").
When depth ordering is resolved correctly the localisation is excellent; it is
usually not attempted.

Caveats: n=47, so 6.38% is 3/47 and the interval is wide. The lexicon also produces
roughly 5 false positives here ("writing on back", "long sleeved" matching 'side'),
which if anything makes the true depth-query accuracy lower still.

Full audit: results/spatial_audit.txt

## 2026-08-12 — Seed variance (jobs 29301 generation, 29306 execution)

Greedy decoding is deterministic, so variance was estimated by sampling at T=0.7,
top_p=0.95, seeds 1-3, RefCOCO/testA, n=500, grounding prompt. Greedy remains the
headline setting because that is what the paper uses for Codex.

| Seed | Accuracy | Mean IoU | spatial | non-spatial |
|---|---:|---:|---:|---:|
| 1 | 34.60% | 0.3476 | 30.45% | 40.28% |
| 2 | 35.60% | 0.3515 | 31.49% | 41.23% |
| 3 | 38.00% | 0.3786 | 36.68% | 39.81% |
| mean +- std | 36.07 +- 1.75 | 0.359 +- 0.017 | 32.87 +- 3.35 | 40.44 +- 0.72 |
| greedy (headline) | 39.20% | 0.3856 | 35.29% | 44.55% |

Greedy beats all three sampled seeds, as expected for a task with one correct
program shape.

TWO OBSERVATIONS.

1. Spatial variance is ~4x non-spatial variance (+-3.35 vs +-0.72). Programs handling
   spatial relations are not merely less accurate, they are less STABLE: across seeds
   the same query is sometimes resolved by sorting and sometimes has its spatial
   constraint dropped. This strengthens the case for providing explicit geometric
   primitives rather than relying on the generator to improvise base-Python sorting.

2. The spatial/non-spatial gap holds on every seed (30.45/40.28, 31.49/41.23,
   36.68/39.81) and on greedy (35.29/44.55). It is not a decoding artefact.

Generation-stage stability was also good: parse rate 97.0 / 97.8 / 98.4, collapse
rate 0.0% on all three seeds.

Seed 2 took 2141s against 319s and 507s — 32 timeouts, against 1 each for the other
seeds. Sampling occasionally produces pathological loops. Worth noting for compute
budgeting; no effect on correctness since timeouts score 0 like any other failure.

## 2026-08-12 — Model sweep: 1.5B (job 29321)

Greedy decoding, RefCOCO/testA, n=500, both prompt conditions. Directly comparable
to the 7B greedy headline numbers.

| Metric | 1.5B base | 1.5B grounding | 7B base | 7B grounding |
|---|---:|---:|---:|---:|
| Task-invalid return rate | 95.8% | 0.0% | 98.8% | 0.2% |
| Spatial constraint dropped | 31.5% | 2.1% | 17.3% | 1.7% |
| Parse rate | 99.8% | 100.0% | 99.8% | 96.6% |
| Mean API calls | 2.18 | 1.17 | 2.45 | 1.66 |

THE EFFECT IS NOT SCALE-DEPENDENT. A model 4.7x smaller fails the released prompt in
the same way (95.8% vs 98.8% task-invalid) and is repaired by the same three
sentences (0.0% vs 0.2%). This removes the most obvious objection to Result 1 —
that it is an artefact of one mid-sized open model rather than a property of the
prompt.

Two secondary observations:

1. Parse rate MOVES IN OPPOSITE DIRECTIONS with the contract: 1.5B improves
   (99.8 -> 100.0) while 7B degrades (99.8 -> 96.6). The 1.5B programs are also much
   simpler (1.17 API calls against 7B's 1.66). A plausible reading is that 1.5B
   copies the exemplars closely, which is syntactically safe but may not generalise;
   7B attempts more and occasionally breaks. Execution will test this — if 1.5B
   copies without understanding, its accuracy should trail 7B despite the better
   parse rate.

2. Baseline spatial-constraint drop is nearly twice as high on 1.5B (31.5% vs 17.3%).
   The smaller model discards spatial language more readily when nothing in the
   prompt requires it.

32B pending (needs 75 of 94 shards on one node). Execution of all four new runs to
follow once it completes.

## 2026-08-14 — Model sweep complete (jobs 29321, 29352 generation; 29404 execution)

Greedy, RefCOCO/testA, n=500, both prompts, three model sizes.

| Model | Baseline acc | Grounding acc | Delta |
|---|---:|---:|---:|
| 1.5B | 0.80% | 34.40% | +33.6 |
| 7B | 0.00% | 39.20% | +39.2 |
| 32B | 0.00% | 40.80% | +40.8 |

Static side:

| Model | Task-invalid, base | Task-invalid, grounding |
|---|---:|---:|
| 1.5B | 95.8% | 0.0% |
| 7B | 98.8% | 0.2% |
| 32B | 87.6% | 0.0% |

RESULT 1 IS SCALE-INDEPENDENT. A 32B model still emits 87.6% task-invalid programs
under the released prompt and still scores 0.00%. Three sentences repair it at every
scale. Scaling 1.5B -> 32B (20x parameters) buys 6.4 accuracy points; the prompt
contract buys 33-41. The prompt effect dominates model capacity by roughly 6x.

THE SPATIAL SUBSET DOES NOT RESPOND TO SCALE AT ALL:

| Model | spatial | non-spatial | gap |
|---|---:|---:|---:|
| 1.5B | 34.95% | 33.65% | -1.3 |
| 7B | 35.29% | 44.55% | +9.3 |
| 32B | 34.95% | 48.82% | +13.9 |

Spatial accuracy is 34.95 / 35.29 / 34.95 — flat to within noise, with 1.5B and 32B
identical to two decimal places. Non-spatial rises 33.65 -> 48.82 (+15.2). So a
larger generator writes better programs for everything EXCEPT spatial relations,
where it writes programs that are exactly as wrong.

This is the strongest evidence yet for the dissertation's premise. If the spatial
deficit were a program-synthesis problem, capacity would help — it demonstrably does
for non-spatial queries on the same images with the same perception stack. It does
not help here because the bottleneck is the API: `compute_depth` returns one scalar,
and no amount of model capacity can express a depth relation the interface cannot
represent.

Anomaly: under the released prompt, 32B failed to define execute_command in 304/500
generations against 16/500 for 1.5B. The larger instruction-tuned model is more prone
to answering in prose when the prompt does not demand a function. That is why 32B
scores 0.00% where 1.5B manages 0.80%.

## 2026-08-14 — Depth wrapper: sign bug found and fixed, and a second bottleneck

### Sign bug (would have inverted the entire contribution)

The original depth.py loaded MiDaS via torch.hub (network — hangs on an offline
compute node) and returned the raw median. I then "fixed" it by inverting, on the
assumption that MiDaS returns inverse depth. verify_depth.py caught that this was
also wrong:

| | with inversion | without |
|---|---|---|
| synthetic near vs far | 0.0018 vs 0.0013 (WRONG) | 565.6 vs 758.6 (correct) |
| real images agreeing with size-distance | 8/25 = 32% | 16/25 = 64% |

32% -> 64% on flipping is systematic inversion, not noise. The HF
`Intel/dpt-hybrid-midas` checkpoint exposes `predicted_depth` already oriented so
larger = further, unlike the torch.hub entrypoint. No inversion is applied. Verified
before building anything on top, which is the lesson from the GLIP tensor_inputs bug.

### The second bottleneck: detection is tuned for detection, not selection

Depth-word queries from RefCOCO+ testA, picking the nearest/farthest box by depth:

| GLIP threshold | depth_order correct | oracle (any box hits) | boxes/img |
|---|---:|---:|---:|
| 0.2 | 0/15 = 0% | 11/15 = 73% | 41.3 |
| 0.5 | 2/15 = 13% | 6/15 = 40% | 2.8 |
| 0.7 | 2/4 = 50% | 3/4 = 75% | 2.2 |

At threshold 0.2 the correct box is PRESENT in 73% of cases and depth ordering picks
it 0% of the time. The candidate set contains 41 boxes per image, mostly overlapping
duplicates, and ordering 41 noisy depth estimates is close to random. At 0.7 the set
shrinks to 2.2 boxes and depth_order recovers 50% against an oracle of 75% — two
thirds of what is achievable.

D8 fixed threshold=0.2 because it maximises TOP-BOX accuracy (81.5%), which is the
right criterion when you take the single best detection. It is the wrong criterion
when a program must SELECT among candidates. These are different operating points
and the pipeline currently uses one threshold for both.

This makes the contribution two-part rather than one, and both parts are now
measured rather than assumed:
  1. the API cannot express depth relations -> add depth_order and friends
  2. the candidate set is not calibrated for selection -> NMS / dedup / top-k before
     any ordering primitive can work

## 2026-08-14 — Depth wrapper: sign bug found and fixed, and a second bottleneck

### Sign bug (would have inverted the entire contribution)

The original depth.py loaded MiDaS via torch.hub (network — hangs on an offline
compute node) and returned the raw median. I then "fixed" it by inverting, on the
assumption that MiDaS returns inverse depth. verify_depth.py caught that this was
also wrong:

| | with inversion | without |
|---|---|---|
| synthetic near vs far | 0.0018 vs 0.0013 (WRONG) | 565.6 vs 758.6 (correct) |
| real images agreeing with size-distance | 8/25 = 32% | 16/25 = 64% |

32% -> 64% on flipping is systematic inversion, not noise. The HF
`Intel/dpt-hybrid-midas` checkpoint exposes `predicted_depth` already oriented so
larger = further, unlike the torch.hub entrypoint. No inversion is applied. Verified
before building anything on top, which is the lesson from the GLIP tensor_inputs bug.

### The second bottleneck: detection is tuned for detection, not selection

Depth-word queries from RefCOCO+ testA, picking the nearest/farthest box by depth:

| GLIP threshold | depth_order correct | oracle (any box hits) | boxes/img |
|---|---:|---:|---:|
| 0.2 | 0/15 = 0% | 11/15 = 73% | 41.3 |
| 0.5 | 2/15 = 13% | 6/15 = 40% | 2.8 |
| 0.7 | 2/4 = 50% | 3/4 = 75% | 2.2 |

At threshold 0.2 the correct box is PRESENT in 73% of cases and depth ordering picks
it 0% of the time. The candidate set contains 41 boxes per image, mostly overlapping
duplicates, and ordering 41 noisy depth estimates is close to random. At 0.7 the set
shrinks to 2.2 boxes and depth_order recovers 50% against an oracle of 75% — two
thirds of what is achievable.

D8 fixed threshold=0.2 because it maximises TOP-BOX accuracy (81.5%), which is the
right criterion when you take the single best detection. It is the wrong criterion
when a program must SELECT among candidates. These are different operating points
and the pipeline currently uses one threshold for both.

This makes the contribution two-part rather than one, and both parts are now
measured rather than assumed:
  1. the API cannot express depth relations -> add depth_order and friends
  2. the candidate set is not calibrated for selection -> NMS / dedup / top-k before
     any ordering primitive can work

### The sign, settled properly

The depth sign flipped twice today before being pinned down, and the reason is worth
recording because it cost hours.

I first assumed MiDaS returns inverse depth and inverted it. A SYNTHETIC test scene
then said the inversion was wrong, so I removed it. The synthetic scene was invalid:
it was built from a brightness gradient with a large dark object low in frame, and
the model read the brightness as depth. A test I designed to be unambiguous was
measuring the wrong thing.

The decisive experiment removes both the detector and the synthetic data. Take
RefCOCO+ testA queries containing closest/nearest/farthest, use GROUND-TRUTH boxes
only, and ask whether the annotated target sits at the correct end of the depth
ordering:

| | correct |
|---|---:|
| without inversion | 2/25 = 8% |
| with inversion | 20/25 = 80% |
| chance (3.8 objects/image) | 26% |

8% is far BELOW chance — the signature of a systematically inverted signal, not a
weak one. MiDaS does return inverse depth (larger = closer); the inversion is
correct and is now documented in depth.py with this evidence.

LESSON: synthetic scenes are unreliable for validating perceptual signals, because
the model latches onto whatever cue the synthetic image actually contains (here,
brightness). Validate against real annotations, and do it first.

### Candidate filtering ablation, with the sign correct

RefCOCO+ depth queries, GLIP threshold 0.2, n=31 usable:

| filter | boxes/img | depth_order | oracle | recovered |
|---|---:|---:|---:|---:|
| none | 39.5 | 9.7% | 83.9% | 11.5% |
| nms (IoU 0.5) | 27.4 | 9.7% | 77.4% | 12.5% |
| top-k (k=6, by area) | 5.8 | 22.6% | 61.3% | 36.8% |
| nms + top-k | 5.8 | 22.6% | 67.7% | 33.3% |

Reference point: the grounding-contract pipeline scores 6.38% on the full RefCOCO+
spatial subset with no depth reasoning at all. Depth ordering over a filtered
candidate set reaches 22.6% on the depth-word subset — roughly 3.5x.

NMS alone does nothing (9.7% either way): the duplicates it removes were not what
confused the ordering. Top-k is what matters, cutting 39.5 candidates to 5.8 and
more than doubling accuracy. It also costs oracle (83.9 -> 61.3), so it discards
correct boxes too — nms+topk keeps more oracle (67.7) at identical accuracy, so that
combination is preferred.

The remaining gap (22.6% achieved against 67.7% achievable) is now the target. Likely
contributors: area is a poor stand-in for detection confidence (the GLIP wrapper
discards scores), and median depth over a box mixes object and background pixels.

CONTRIBUTION IS NOW THREE PARTS, ALL MEASURED:
  1. correct depth semantics (sign + per-image normalisation)
  2. candidate filtering before ordering (top-k, ideally by real confidence)
  3. depth_order as an exposed API primitive
