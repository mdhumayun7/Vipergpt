# Reproduction results

Every number below is traceable to a run directory under `outputs/runs/`, a
configuration, and a SLURM job ID. Numbers from a login-node or `-dirty` run are not
cited (see the note on discarded runs at the end).

**Scope.** RefCOCO and RefCOCO+ visual grounding (the paper's Table 1) only. GQA,
OK-VQA and NExT-QA are out of scope — see D2 in `docs/deviations.md`.

**Metric.** Accuracy = fraction of samples whose returned box has IoU ≥ 0.5 with the
RefCOCO ground-truth box, matching the paper's protocol. A program that returns
anything other than an `ImagePatch` scores 0 rather than being excluded: returning a
sentence where a box is required is a task failure, and dropping such cases would
flatter the baseline.

---

## Table 1 — RefCOCO / RefCOCO+, testA, n = 500 per cell

| Condition | RefCOCO | RefCOCO+ |
|---|---:|---:|
| ViperGPT, **as reported in the paper** (Codex) | **72.0** | **67.0** |
| Reproduced, **released prompt** (Qwen2.5-Coder-7B) | **0.00** | **0.00** |
| Reproduced, **+ grounding contract** (this work) | **39.20** | **31.80** |

Mean IoU, same runs:

| Condition | RefCOCO | RefCOCO+ |
|---|---:|---:|
| Released prompt | 0.0000 | 0.0000 |
| + grounding contract | 0.3856 | 0.3245 |

Provenance:

| Cell | Run directory | Job |
|---|---|---|
| RefCOCO baseline | `20260812T115440__nogit__m2_baseline_full` | 29290 |
| RefCOCO grounding | `20260812T115836__nogit__m2_grounding_full` | 29290 |
| RefCOCO+ baseline | `20260812T121740__nogit__m2_baseline_plus` | 29292 |
| RefCOCO+ grounding | `20260812T122250__nogit__m2_grounding_plus` | 29292 |

`git_sha` reads `nogit` because `git` is not installed on the compute nodes. The
SLURM job IDs and the committed job scripts (`run_m2.sh`, `run_m2_plus.sh`) provide
traceability instead. Fixing this requires capturing the SHA on the login node at
submission and passing it through the environment.

---

## Why the released prompt scores exactly zero

Not a crash, and not a configuration artefact. Under the released prompt on RefCOCO,
336 of 500 programs executed to completion and returned a **string**. Zero returned
an `ImagePatch`. Grounding is scored by IoU against a box, so a sentence cannot
score, however correct it is as an English answer.

| Return type | RefCOCO baseline | RefCOCO grounding |
|---|---:|---:|
| `ImagePatch` | 0 | 443 |
| `str` | 336 | 1 |
| error | 163 | 56 |
| `bool` | 1 | 0 |

The prompt never states that a grounding program must return a patch, and its
in-context examples return strings (`simple_query`, `best_text_match`, `llm_query`).
Codex evidently inferred the contract; an open-weights generator does not. Adding
three sentences and two grounding exemplars is the entire intervention.

Missing modules were ruled out as a cause: `simple_query` and `llm_query` are given
neutral empty-string fallbacks (D10) so no program can fail for lack of a module we
chose not to load. This concession strictly favours the baseline and does not change
its score.

### A second, unanticipated effect

Under the released prompt a large fraction of generations never defined
`execute_command` at all — the model emitted prose or malformed markdown:

| Dataset | Released prompt | + grounding contract |
|---|---:|---:|
| RefCOCO | 136 / 500 (27.2%) | 0 |
| RefCOCO+ | 214 / 500 (42.8%) | 0 |

The contract disciplines output *format* as well as return type. These are separable
failures and are reported separately.

---

## Where the remaining gap comes from

Reproduced 39.20 against a reported 72.0 — a 32.8-point shortfall. The known
contributors, in descending order of likely size:

| # | Contributor | Evidence |
|---|---|---|
| D1 | Codex → Qwen2.5-Coder-7B-Instruct | Codex is retired; no faithful option exists. Program quality is the dominant term. |
| — | Perception ceiling | Raw GLIP top-box accuracy on this data is 78.9% (n=20, threshold 0.2). Program-level selection loses ~40 points from that ceiling. |
| D9 | X-VLM → CLIP for `verify_property` | Most costly on RefCOCO+, whose expressions are attribute-based. RefCOCO+ scores 31.80 against RefCOCO's 39.20. |
| D8 | GLIP threshold 0.2, tuned on 20 samples | Materially affects results and is under-validated. |
| D10 | `simple_query` / `llm_query` return `""` | Favours the baseline; no measurable effect on the grounding condition (39.20 either way). |

The gap is not attributed to any single cause and no claim of a faithful reproduction
is made. What is reproduced is the *method*; what is measured is what happens to it
when its generator is replaced.

---

## Table 2 — Spatial reasoning breakdown

Queries were split by a relational-term lexicon (`left`, `behind`, `closest`,
`between`, …). Grounding-contract condition only, since the baseline scores 0
everywhere.

| Dataset | Subset | n | Accuracy | Mean IoU |
|---|---|---:|---:|---:|
| RefCOCO | spatial | 289 | 35.29 | 0.3475 |
| RefCOCO | non-spatial | 211 | 44.55 | 0.4378 |
| RefCOCO+ | spatial | 47 | **6.38** | 0.1037 |
| RefCOCO+ | non-spatial | 453 | 34.44 | 0.3474 |

A manual audit of all 47 RefCOCO+ spatial queries (`results/spatial_audit.txt`)
shows the two spatial subsets are **not the same kind of query**:

| | RefCOCO | RefCOCO+ |
|---|---|---|
| Spatial language | 2D image-plane | depth and inter-object |
| Examples | "man on right", "person bottom left" | "man closest to us", "farthest man", "umpire behind catcher" |
| Resolvable by | sorting patch coordinates in base Python | requires `compute_depth` or 3D geometry |
| Accuracy | 35.29 | 6.38 |

RefCOCO+ forbids spatial language, which removed viewer-frame *directional*
expressions but left depth-ordering and relational ones. The released prompt
explicitly instructs the generator to use "base Python (comparison, sorting) for
left/right/up/down", which works for the 2D case and cannot work for the depth case.
Milestone 1 found `compute_depth` invoked 29 times against 547 for `find`.

All three successes in the RefCOCO+ spatial subset are depth queries with high IoU
(0.97, 0.96, 0.92): when depth ordering is attempted correctly the localisation is
excellent, but it is usually not attempted.

Caveats: n = 47, so 6.38% is 3 of 47 and the interval is wide. The lexicon also
produces roughly five false positives in that subset ("writing on back", "long
sleeved" matching *side*), which would push true depth-query accuracy lower still.

---

## Discarded runs

An earlier 500-sample execution was run on the login node, where no GPU is visible.
GLIP raised `RuntimeError` 481 times and the "grounding" run completed in 7 seconds,
yielding a complete and entirely fake 0.00% / 0.00% table. Those directories were
deleted and the job script now asserts `torch.cuda.is_available()` and aborts
otherwise. Silent CPU fallback is the most dangerous failure mode encountered in this
project: it produces plausible output rather than an error.

---

## Not yet done

- Multiple seeds. All results are single-run, greedy decoding. Mean ± std over
  at least three seeds is required before these numbers are final.
- GLIP threshold re-swept on a larger sample than 20 (D8).
- X-VLM proper, replacing the CLIP substitute (D9).
- Generator sweep (1.5B / 32B) to test whether the prompt effect is scale-dependent.
