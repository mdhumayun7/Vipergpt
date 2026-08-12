# Reproduction results

Every number is traceable to a run directory under `outputs/runs/`, an archived
summary in `results/m2_summaries/`, and a SLURM job ID.

**Scope.** RefCOCO / RefCOCO+ visual grounding (the paper's Table 1) only. GQA,
OK-VQA and NExT-QA are out of scope — see D2 in `docs/deviations.md`.

**Metric.** Accuracy = fraction of samples whose returned box has IoU >= 0.5 with the
RefCOCO ground-truth box. A program returning anything other than an `ImagePatch`
scores 0 rather than being excluded: returning a sentence where a box is required is
a task failure, and dropping such cases would flatter the baseline.

---

## Table 1 — RefCOCO / RefCOCO+, testA, n = 500 per cell

| Condition | RefCOCO | RefCOCO+ |
|---|---:|---:|
| ViperGPT, **as reported** (Codex) | **72.0** | **67.0** |
| Reproduced, **released prompt** (Qwen2.5-Coder-7B, greedy) | **0.0** | **0.0** |
| Reproduced, **+ grounding contract** (greedy) | **39.2** | **31.8** |
| Reproduced, + grounding contract (T=0.7, 3 seeds) | 36.07 +- 1.75 | not run |

Mean IoU: released prompt 0.0 / 0.0;
grounding contract 0.3856 / 0.3245.

| Cell | Run directory | Job |
|---|---|---|
| RefCOCO baseline | `20260812T115440__nogit__m2_baseline_full` | 29290 |
| RefCOCO grounding | `20260812T115836__nogit__m2_grounding_full` | 29290 |
| RefCOCO+ baseline | `20260812T121740__nogit__m2_baseline_plus` | 29292 |
| RefCOCO+ grounding | `20260812T122250__nogit__m2_grounding_plus` | 29292 |
| seeds 1-3 | `...__m2_grounding_s1/s2/s3` | 29306 |

`git_sha` reads `nogit` because git is not installed on the compute nodes; SLURM job
IDs and the committed job scripts provide traceability instead.

---

## Why the released prompt scores exactly zero

Not a crash. On RefCOCO, 336 of 500 programs ran to completion and
returned a **string**; zero returned an `ImagePatch`. Grounding is scored by IoU
against a box, so a sentence cannot score however correct its English.

| Return type | RefCOCO baseline | RefCOCO grounding |
|---|---:|---:|
| `ImagePatch` | 0 | 443 |
| `str` | 336 | 1 |
| error | 163 | 56 |

The released prompt never states that a grounding program must return a patch, and
its in-context examples return strings. Codex evidently inferred the contract; an
open-weights generator does not. Three sentences and two exemplars are the entire
intervention.

Missing modules are ruled out: `simple_query` and `llm_query` get neutral
empty-string fallbacks (D10), so no program fails for lack of a module we chose not
to load. That concession favours the baseline and does not change its score.

### Second effect: output format

Generations that never defined `execute_command` at all:

| Dataset | Released prompt | + grounding contract |
|---|---:|---:|
| RefCOCO | 136 / 500 | 0 |
| RefCOCO+ | 214 / 500 | 0 |

The contract disciplines output *format* as well as return type. Separable failures,
reported separately.

---

## Where the remaining gap comes from

Reproduced 39.2 against a reported 72.0.

| # | Contributor | Evidence |
|---|---|---|
| D1 | Codex -> Qwen2.5-Coder-7B | Codex retired; no faithful option. Dominant term. |
| — | Perception ceiling | Raw GLIP top-box accuracy 78.9% (n=20, thr 0.2). Program selection loses ~40 points from there. |
| D9 | X-VLM -> CLIP | Costliest on RefCOCO+, whose expressions are attribute-based. |
| D8 | GLIP threshold tuned on 20 samples | Material and under-validated. |
| D10 | text calls return "" | Favours the baseline; no effect on grounding. |

No claim of a faithful reproduction is made. What is reproduced is the method; what
is measured is what happens to it when its generator is replaced.

---

## Table 2 — Spatial reasoning breakdown

Grounding-contract condition (baseline scores 0 everywhere). Split by a
relational-term lexicon.

| Dataset | Subset | n | Accuracy | Mean IoU |
|---|---|---:|---:|---:|
| RefCOCO | spatial | 289 | 35.29 | 0.3475 |
| RefCOCO | non-spatial | 211 | 44.55 | 0.4378 |
| RefCOCO+ | spatial | 47 | **6.38** | 0.1037 |
| RefCOCO+ | non-spatial | 453 | 34.44 | 0.3474 |

A manual audit of all 47 RefCOCO+ spatial queries (`results/spatial_audit.txt`) shows
the two spatial subsets are **not the same kind of query**:

| | RefCOCO | RefCOCO+ |
|---|---|---|
| Spatial language | 2D image-plane | depth and inter-object |
| Examples | "man on right", "person bottom left" | "man closest to us", "farthest man", "umpire behind catcher" |
| Resolvable by | sorting patch coordinates in base Python | requires `compute_depth` or 3D geometry |
| Accuracy | 35.29 | 6.38 |

RefCOCO+ forbids spatial language, which removed viewer-frame *directional*
expressions but left depth-ordering and relational ones. The released prompt
instructs the generator to use "base Python (comparison, sorting) for
left/right/up/down" — which works for the 2D case and cannot work for the depth case.
Milestone 1 found `compute_depth` invoked 29 times against 547 for `find`.

All three successes in the RefCOCO+ spatial subset are depth queries with high IoU
(0.97, 0.96, 0.92): when depth ordering is attempted correctly, localisation is
excellent; it is usually not attempted.

Caveats: n=47, so 6.38% is 3 of 47 and the interval is wide. The lexicon also
yields ~5 false positives in that subset, which would push true depth-query accuracy
lower still.

### Seed variance (T=0.7, seeds 1-3, RefCOCO)

| Subset | mean +- std |
|---|---:|
| overall | 36.07 +- 1.75 |
| spatial | 32.87 +- 3.34 |
| non-spatial | 40.44 +- 0.72 |

Spatial variance is ~4x non-spatial variance. Programs handling spatial relations are
not merely less accurate but less stable: across seeds the same query is sometimes
resolved by sorting and sometimes has its spatial constraint dropped.

---

## Discarded runs

An earlier 500-sample execution ran on the login node, where no GPU is visible. GLIP
raised `RuntimeError` 481 times and the "grounding" run finished in 7 seconds,
yielding a complete and entirely fake 0.00 / 0.00 table. Those directories were
deleted; job scripts now assert `torch.cuda.is_available()` and abort otherwise.
Silent CPU fallback is the most dangerous failure mode encountered here — it produces
plausible output rather than an error.

---

## Not yet done

- GLIP threshold re-swept on more than 20 samples (D8).
- X-VLM proper, replacing the CLIP substitute (D9).
- Generator sweep (1.5B / 32B) to test whether the prompt effect is scale-dependent.
- RefCOCO+ seed variance (only RefCOCO was run across seeds).
