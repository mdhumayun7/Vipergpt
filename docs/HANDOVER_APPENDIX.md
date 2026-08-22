# Project Handover — Appendix

**Paste this immediately after `PROJECT_HANDOVER.md`.** It contains the details that
are easy to get wrong and expensive to get wrong.

---

## A. Protocol constants — use these exact values

| Parameter | Value | Note |
|---|---|---|
| Split | `testA` | both datasets |
| Samples per cell | **500** | first 500 **in file order**, not a seeded shuffle |
| Decoding (headline) | greedy, `temperature=0.0` | matches the paper's temperature 0 |
| Decoding (variance) | `temperature=0.7, top_p=0.95`, seeds 1–3 | never mixed with greedy figures |
| `max_new_tokens` | **320** | paper specifies 512 — an unrecorded difference, see §I |
| Executor timeout | **300 s** | was 60 s; raised after 40 RefCOCO+ timeouts under C4 |
| IoU threshold | **0.5** | paper's Table 1 protocol |
| GLIP threshold | **0.2** | D8, validated held-out on samples 200–399 |
| `max_detections` | **0 (off)** | D12 — enabling it inflates results, see the control |
| `find_nms_iou` | **0 (off)** | harms RefCOCO+ depth queries |
| `crop_larger_margin` | **False** | paper sets true — D13, effect unmeasured |
| Batch size | 8 (Qwen-7B, Yi, OpenCoder) · 4 (Qwen-32B) · **1–2 (DeepSeek)** | D17 |

**The paper's own sampling differs.** `paper_spec.md §5` records
`np.random.seed(4)` before a shuffle, and the paper reports on the **full** testA
split. This work takes the first 500 in file order. Both are deterministic; they are
not the same 500 samples. State this whenever a number is compared to 72.0.

---

## B. The spatial lexicon — load-bearing for every subset claim

A query is labelled **spatial** if it contains any of:

```
left, right, behind, in front, front, above, below, under, underneath,
on top, top, beneath, next to, beside, near, nearest, closest, farthest,
closer, between, around, middle, center, corner, side, back, far, facing,
over, across
```

Matched as whole words, case-insensitive. Everything else is **non-spatial**.

**This is a heuristic, not annotation, and it is imperfect.** The manual audit
(`results/spatial_audit.txt`) found roughly five false positives among the 47
RefCOCO+ spatial queries — *"black shirt writing on back"*, *"man with glasses in
long sleeved..."* matching *side*. Always state this when quoting a subset figure.

**Subset sizes:** RefCOCO 289 spatial / 211 non-spatial. RefCOCO+ **47** spatial /
453 non-spatial.

**The two spatial subsets are not the same kind of query** — this is the finding that
motivated Milestone 3:

| | RefCOCO | RefCOCO+ |
|---|---|---|
| Language | 2D image-plane | depth and inter-object |
| Examples | *man on right*, *person bottom left* | *man closest to us*, *farthest man*, *umpire behind catcher* |
| Resolvable by | sorting patch coordinates | requires `compute_depth` or 3D geometry |

RefCOCO+ forbids *location* words, which removed viewer-frame directional
expressions and left depth ones. That is why the gap **widened** on RefCOCO+ rather
than narrowing — a rejected hypothesis that produced the contribution.

---

## C. Coordinate convention — the paper's own bug trap

`ImagePatch` uses a **bottom-left origin**: `(left, lower, right, upper)`.
The detector returns **top-left origin**. The conversion happens in `glip.py` on
return, and again in `execute_refcoco.py` before scoring:

```python
pred = (r.left, H - r.upper, r.right, H - r.lower)   # undo the flip for COCO
iou  = iou_xyxy(pred, g["gt_xyxy"])
```

`paper_spec.md §5` flags this explicitly: get it wrong and **IoU silently collapses**.
It was verified empirically — flipped gives mean IoU 0.776, unflipped 0.364.

**Two related failures already hit, both silent:**

- `GLIPDemo` defaults to `tensor_inputs=False`, under which it reads the image size
  as `shape[:-1]`, assuming HWC. Passing a CHW tensor makes it read `(3, H)` and
  rescale every box into a three-pixel-wide image. Every IoU came out **exactly**
  zero — not low, zero — which is what identified it as a coordinate bug rather than
  a detection-quality problem.
- Child patches from `find()` were not offset into the parent's frame. Invisible for
  root-level programs, wrong for anything that crops first.

**Rule of thumb learned here:** an *exactly* zero metric is a coordinate or type bug;
a *low* metric is a quality problem. They are diagnosed differently.

---

## D. Figure inventory — 18 figures, what each answers

| # | File | Shows |
|---|---|---|
| 1 | `fig1_main_result` | C1–C4 × both datasets × both subsets — the main table as bars |
| 2 | `fig2_scale` | spatial flat / non-spatial rising across 1.5B–32B. **The load-bearing figure.** |
| 3 | `fig3_api_adoption` | call counts per condition — did the generator use the new API |
| 4 | `fig4_qualitative` | 3 depth queries where C4 succeeds and C2 fails, with boxes |
| 5 | `fig5_iou_distribution` | IoU distribution (bimodal) + share scoring exactly zero |
| 6 | `fig6_seed_variance` | mean ± s.d. and the 4.6× spatial stability gain |
| 7 | `fig7_failure_modes` | every sample classified — separates specification bug from perception limit |
| 8 | `fig8_threshold_tradeoff` | one threshold, two incompatible jobs (detection vs selection) |
| 9 | `fig9_depth_sign` | 8% vs 26% chance vs 80% — the sign evidence |
| 10 | `fig10_vs_paper` | reproduced vs reported, with the 81.5 perception ceiling |
| 11 | `fig11_iou_sweep` | C4−C2 gap stays positive across IoU 0.3–0.9 |
| 12 | `fig12_headroom` | achieved vs oracle — ordering failure vs detection failure |
| 13 | `fig13_cap_control` | the control that disqualified the 15-point result |
| 14 | `fig14_architecture` | pipeline, colour-coded by provenance |
| 15 | `fig15_paper_vs_ours` | component-by-component design comparison |
| 16 | `fig16_paper_vs_result` | reproduction vs paper's Table 1 incl. its zero-shot baselines |
| 17 | `fig17_grid` | four models × four conditions × both datasets |
| 18 | `fig18_generalisation` | three panels: does each finding transfer, before/after per model |

All regenerate from archived summaries. **No figure contains a typed number.**

---

## E. Dissertation chapter map

`docs/dissertation.tex`, single file, ~50 pages, compiles clean with
`pdflatex → bibtex → pdflatex → pdflatex` **from the repository root** (figure paths
are `results/figures/*.pdf`).

| # | Chapter |
|---|---|
| 1 | Introduction |
| 2 | Literature Review |
| 3 | Problem Formulation |
| 4 | System Architecture and Methodology (incl. full C1–C4 specification) |
| 5 | Implementation (code listings, GLIP build, depth backend) |
| 6 | Experimental Setup (incl. threshold sweep) |
| 7 | Results |
| 8 | Rejected Hypotheses and Controls |
| **9** | **Milestone 4 — Generalisation Across Generator Families** |
| 10 | Deviations from the Published Configuration |
| 11 | Limitations |
| 12 | Conclusion and Future Work |

**Known gap:** Chapters 1 (§Approach, §Contributions) and 12 still describe **three**
milestones. Milestone 4 needs adding to both. Not yet done.

Second document: `docs/viper_c1c4_notes.tex` — 10 pages, professor-facing, standalone
(TikZ diagrams, no external figures needed).

---

## F. What must NOT be claimed

This list exists because two attractive results were already rejected under it.

1. **Not a faithful reproduction.** 42.00 (or 45.40 with Yi) against 72.0. Never say
   "reproduced ViperGPT" without qualification.
2. **Not metric-equivalent to the paper** unless stated — see §A and the metric
   ambiguity note in the main handover.
3. **No contribution from the candidate cap.** The control shows C2 gains +22.5
   spatial points from it with no depth primitives present. **57.60 must never appear
   as a result.**
4. **No metric depth, no 3D positions in metres.** COCO ships no intrinsics.
   `position_3d` and `physical_size` exist in code but rest on an assumed 60° FOV and
   were used in **no** reported number. Claim **ordering only**.
5. **No claim on GQA, OK-VQA or NExT-QA.** Never run.
6. **No X-VLM result.** `_xvlm_backbone.py` raises `NotImplementedError`.
7. **No statistical significance on the depth subset.** n=47; the seed s.d. is
   literally ±1 sample. Direction only.
8. **Prototype selection-study numbers are not end-to-end results.**
   `results/score_filter_ablation.json` (29.0 / 25.8 / 29.0 / 32.3) comes from
   oracle-style picking, not generated programs.
9. **No novel algorithm.** The method is the paper's. What is new is the API surface,
   the prompt specification, and the measurement.
10. **"C4 beats C3" is not universal.** True for every generator on RefCOCO; on
    RefCOCO+ DeepSeek is a −0.2 exception, unconfirmed against re-seeding.

---

## G. Observations that are real but unexplained — do not over-interpret

1. **DeepSeek's C2 is anomalously weak** (29.60 RefCOCO, 24.00 RefCOCO+, against
   39–44 for the others). Consequently its apparent C3 "gain" of +14.2 on non-spatial
   is better read as *C2 being poor* than as depth primitives helping non-spatial
   queries. Consistent with 2.4B active parameters.

2. **OpenCoder and Yi score far higher than Qwen on RefCOCO+ depth queries under C2
   already** — 23.40 and 25.53 against Qwen's 4.26, *before* any depth primitive is
   offered. Some generators evidently resolve depth relations by another route under
   the plain contract. **Not investigated.** This is a genuinely open question and a
   good thread to pull.

3. **Three models land on exactly 19.15 on RefCOCO+ spatial under C4.** That is
   9/47 — a consequence of the 2.13-point quantisation at n=47, not meaningful
   agreement.

4. **OpenCoder loses 6.4 points on the depth subset from C2→C3** while the other
   three gain. Unexplained.

5. **21 of 200 samples yield no detection at every threshold up to 0.4.** Not
   threshold-related — it comes from the head-noun heuristic used to prompt GLIP
   (*"player number 8"* → *"8"* → discarded by an `isalpha` filter). A separate,
   known limitation, and relevant to the C5 plan below.

---

## H. C5 — draft, ready to build

**Hypothesis.** Generated programs pass **bare nouns** to `find()` and then filter
with CLIP, which is the weak substitute (D9). GLIP is a *grounded* detector and
accepts full phrases. Giving it the descriptive phrase should improve RefCOCO+
directly, where accuracy is weakest.

**Confirm first** (see §10 of the main handover for the exact diagnostic). Mean words
per `find()` argument ≈ 1.0 confirms it.

**Build:** `prompts/api_depth_desc.prompt` = C4 + the following. Nothing else changes,
so any difference is attributable to this text alone.

```
- Pass the DESCRIPTIVE PHRASE to find(), not the bare noun. The detector is a
  grounded detector and understands attributes and modifiers directly.
  Prefer  find("man in a red jacket")  over  find("man")  followed by
  verify_property. Use verify_property only to disambiguate among the
  detections that find() returns, not to do the finding.
- Keep the phrase short and visual. Drop relational clauses that the detector
  cannot see: for "the man in a red jacket closest to us", call
  find("man in a red jacket") and then depth_order(...)[0].
```

Plus two examples:

```python
>>> # Query: the woman in the red jacket
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     women = image_patch.find("woman in a red jacket")
>>>     if not women: return image_patch
>>>     return women[0]

>>> # Query: the man in the blue shirt closest to the camera
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     men = image_patch.find("man in a blue shirt")
>>>     if not men: return image_patch
>>>     return depth_order(men)[0]
```

**Why this fits the existing story.** C2 said *what to return*. C4 said *when a
capability applies*. C5 says *how much of the query to give the detector*. All three
are failures of specification rather than of the model — the dissertation's thesis.

**Risk to check:** a longer phrase may return **zero** detections where a bare noun
returned many. Measure the empty-detection rate alongside accuracy, or the gain will
be misread. There is already a 10.5% floor of no-detection cases (§G.5).

---

## I. Open items and unresolved mismatches

| | Item | Status |
|---|---|---|
| 1 | X-VLM (D9) never built | open, 2–3 days |
| 2 | Qwen-1.5B / 32B not in the full grid (12 cells) | open, low priority |
| 3 | Grid cells are single-seed | open |
| 4 | `crop_larger_margin` effect unmeasured (D13) | open |
| 5 | `max_new_tokens=320` vs the paper's 512 | **not yet in the deviation register** — should be D19 |
| 6 | Milestone 4 missing from Ch. 1 and Ch. 12 | open, quick |
| 7 | `configs/default.yaml` describes no real run (D14) | documented, not fixed |
| 8 | DeepSeek C3/C4 −0.2 on RefCOCO+ | reported, not re-seeded |
| 9 | Why do OpenCoder/Yi handle depth better under C2? (§G.2) | unexamined, interesting |

**Item 5 is a real gap.** The paper specifies `max_tokens=512`; this work uses 320.
It has not been recorded as a deviation and should be, with a note on whether any
program was truncated by it. Worth checking before the next report goes out:

```bash
python -c "
import json, glob
d = sorted(glob.glob('outputs/runs/*m1_refcoco_testA_7B_depth_cond_greedy'))[-1]
r = [json.loads(l) for l in open(d+'/programs.jsonl')]
long = [x for x in r if not x['program'].rstrip().endswith((')',']','\"',\"'\",'0','1','2','3','4','5','6','7','8','9'))]
print('programs not ending in a plausible terminal token:', len(long), 'of', len(r))
for x in long[:3]: print('---'); print(x['program'][-160:])
"
```

If several programs end mid-expression, truncation is real and 320 is too low.
