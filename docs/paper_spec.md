# Paper Spec — ViperGPT

**Paper:** ViperGPT: Visual Inference via Python Execution for Reasoning
**Authors:** Dídac Surís\*, Sachit Menon\*, Carl Vondrick (Columbia University)
**Venue / year:** ICCV 2023; preprint arXiv:2303.08128 (**v1, 14 Mar 2023** — the version read)
**Project / code:** https://viper.cs.columbia.edu · https://github.com/cvlab-columbia/viper
**Reproduction targets:** Tables 1, 2, 3, 4 (RefCOCO/RefCOCO+, GQA, OK-VQA, NExT-QA).

This document is what we implement against and becomes the "Method" chapter of the
thesis. It is written from the paper plus the official repository (which is treated
as ground truth for under-specified implementation details).

---

## 1. Problem statement

Given a visual input `x` (an image or a video) and a natural-language query `q`,
produce a result `r` that answers the query. ViperGPT does this in two steps:

1. **Program generation:** a code-generation LLM `π` maps the query to a Python
   program: `z = π(q)`. `z` is a string defining `def execute_command(image[, ...])`.
2. **Program execution:** an execution engine `φ` runs the program on the input:
   `r = φ(x, z)`, calling pretrained vision/language modules through a fixed API.

The framework is **training-free** (zero-shot): no gradients, no fine-tuning.

**Inputs / outputs / shapes.**

| Symbol | Meaning | Type / shape |
|--------|---------|--------------|
| `x` (image) | RGB image tensor | `float[3, H, W]`, channel-first, values as loaded by torchvision |
| `x` (video) | ordered frames | `float[T, 3, H, W]` |
| `q` | query string | `str` |
| `z` | generated program | `str` (Python source) |
| `r` | result | one of: `str` (answer), `ImagePatch` / bbox (grounding), `int`/`bool` (multiple-choice index / logical) |

`ImagePatch` stores `cropped_image (float[3,h,w])`, integer box `left/lower/right/upper`,
plus derived `width, height, horizontal_center, vertical_center`.
`VideoSegment` stores the frame stream, its start/end timestamps, and yields
`ImagePatch` objects via a frame iterator.

---

## 2. Full pipeline (stages, in order, with boundary shapes)

```
q (str)                          x (image[3,H,W] or video[T,3,H,W])
  │                                  │
  ▼                                  │
[S1] Prompt assembly                 │
  API spec prompt + query  ──► prompt (str)
  │                                  │
  ▼                                  │
[S2] Program generation π            │
  code LLM(prompt) ──► z (str: "def execute_command(...)")
  │                                  │
  ▼                                  ▼
[S3] Program compilation           input wrapped as ImagePatch(x) / VideoSegment(x)
  exec(z) ──► callable execute_command
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
[S4] Program execution φ  (Python interpreter + API implementation)
      calls modules on demand:
        find / exists            ──► GLIP           (patch → List[bbox])
        verify_property          ──► X-VLM          (patch, obj, prop → bool)
        best_image/text_match    ──► X-VLM / CLIP   (patches, text → patch/str)
        simple_query             ──► BLIP-2         (patch, question → str)
        compute_depth            ──► MiDaS          (patch → float)
        llm_query / select_answer──► text LLM        (text → str)
        distance/sort/math/logic ──► base Python
                 │
                 ▼
[S5] Result post-processing / normalization ──► r
                 │
                 ▼
[S6] Metric evaluation (per dataset) ──► score
```

Tensor shapes at boundaries: `ImagePatch.cropped_image` is `float[3, h, w]`;
`find` returns `List[ImagePatch]`; `compute_depth` returns a scalar `float`
(median of the crop's relative depth map); module text I/O is plain `str`.

---

## 3. Key definitions and the module API

The program generator is conditioned on an **API specification prompt** (no
implementations — only signatures, docstrings, and few-shot query→code examples).
The exact API (from `prompts/chatapi.prompt`) defines:

**Class `ImagePatch`** with methods:
- `find(object_name) -> List[ImagePatch]`
- `exists(object_name) -> bool`  (≡ `len(find(object_name)) > 0`)
- `verify_property(object_name, visual_property) -> bool`
- `best_text_match(option_list, prefix=None) -> str`
- `simple_query(question=None) -> str`  (default question "What is this?")
- `compute_depth() -> float`  (median relative depth of the crop)
- `crop(left, lower, right, upper) -> ImagePatch`
- `overlaps_with(left, lower, right, upper) -> bool`
- `llm_query(question, long_answer=True) -> str`
- attributes: `left, lower, right, upper, width, height, horizontal_center, vertical_center`

**Module functions:**
- `best_image_match(list_patches, content, return_index=False) -> ImagePatch | int`
- `distance(patch_a, patch_b) -> float` (edge distance; negative = −IoU if overlapping)
- `bool_to_yesno(bool) -> str`
- `coerce_to_numeric(string) -> float`

**Class `VideoSegment`** (video tasks): holds `video` bytestream + `start`/`end`
timestamps; `frame_iterator()` yields per-frame `ImagePatch`; `frame_from_index`
+ `trim` for temporal windows; and `select_answer(info, question, options)` for
multiple-choice (via a text LLM). *(NExT-QA-specific; see §5.)*

**Program-generation formalism.** `z = π(q)` where `π` is an autoregressive code
LLM. In the paper `π` = Codex `code-davinci-002` (temperature 0). No custom
interpreter: `z` is standard Python run by `exec`, so it can freely use `for`,
`if/else`, `sort`, `math`, list ops, etc. This "faithfulness by construction" is
the paper's core claim.

There are no trained loss functions or equations to optimize; the "algorithm" is
program synthesis + interpretation. The only numeric thresholds are module
decision thresholds (§7).

---

## 4. Datasets

| Task | Dataset | Split reported | Metric | Notes / licence |
|------|---------|----------------|--------|-----------------|
| Visual grounding (T1) | **RefCOCO** & **RefCOCO+** | **testA** | IoU / [email protected] (**see OQ-1**) | COCO images; UNC splits; research licence |
| Compositional VQA (T2) | **GQA** | **test-dev (balanced)** | exact-match accuracy | GQA licence; images from Visual Genome/COCO |
| Knowledge VQA (T3) | **OK-VQA** | test/val (**see OQ-2**) | soft VQA accuracy (10-annotator) | COCO images; CC licence |
| Video reasoning (T4) | **NExT-QA** | **val, multiple-choice**; Hard split (T/C) + Full set | MC accuracy | NExT-QA licence; videos from VidOR |

Preprocessing: images loaded as channel-first RGB float tensors; GLIP/BLIP-2/X-VLM
apply their own resizing/normalization internally. For video, NExT-QA clips are
decoded to frames and wrapped in `VideoSegment`. Exact frame-sampling stride is an
open question (**OQ-3**).

**Reported numbers to reproduce (transcribed from the paper).**

*Table 1 — RefCOCO / RefCOCO+, IoU (%), testA. Zero-shot rows; supervised shown for context.*

| Method | RefCOCO | RefCOCO+ |
|--------|--------:|---------:|
| MDETR (sup.) | 90.4 | 85.5 |
| OFA (sup.) | 94.0 | 91.7 |
| OWL-ViT (ZS) | 30.3 | 29.4 |
| GLIP (ZS) | 55.0 | 52.2 |
| ReCLIP (ZS) | 58.6 | 60.5 |
| **ViperGPT (ZS)** | **72.0** | **67.0** |

*Table 2 — GQA test-dev, accuracy (%).*

| Method | Accuracy |
|--------|---------:|
| LGCN / LXMERT / NSM / CRF (sup.) | 55.8 / 60.0 / 63.0 / 72.1 |
| BLIP-2 (ZS) | 44.7 |
| **ViperGPT (ZS)** | **48.1** |

*Table 3 — OK-VQA, accuracy (%).*

| Method | Accuracy |
|--------|---------:|
| TRiG / KAT / RA-VQA / REVIVE / PromptCap (sup.) | 50.5 / 54.4 / 54.5 / 58.0 / 58.8 |
| PNP-VQA / PICa / BLIP-2 / Flamingo (ZS) | 35.9 / 43.3 / 45.9 / 50.6 |
| **ViperGPT (ZS)** | **51.9** |

*Table 4 — NExT-QA, accuracy (%).*

| Method | Hard-T | Hard-C | Full |
|--------|-------:|-------:|-----:|
| ATP (sup.) | 45.3 | 43.3 | 54.3 |
| VGT (sup.) | – | – | 56.9 |
| HiTeA (sup.) | 48.6 | 47.8 | 63.1 |
| **ViperGPT (ZS)** | **49.8** | **56.4** | **60.0** |

---

## 5. Evaluation protocol (exact)

- **RefCOCO/RefCOCO+ (T1):** the program returns an `ImagePatch` (bbox). **OQ-1
  RESOLVED** from official `datasets/refcoco.py`: the evaluator returns *two*
  numbers — **mean IoU** (primary; this is the reported 72.0/67.0) and a secondary
  **[email protected]** (`iou > 0.7`). A `None` prediction falls back to the mean box
  (≈22.64% IoU). **Coordinate convention (bug trap):** the dataset flips y to a
  bottom-left origin, so boxes are `(left, lower, right, upper)` and `box_iou`
  expects exactly that order — get this wrong and IoU silently collapses.
  Sampling uses `np.random.seed(4)` before shuffle, so subsets are deterministic.
- **GQA (T2):** open-ended answer string compared to the single GQA gold answer with
  GQA's answer normalization; report exact-match accuracy over test-dev balanced.
- **OK-VQA (T3):** standard VQA soft accuracy: `acc = min(1, (#matching human
  answers)/3)`, averaged over the 10 annotations and over questions, after VQA
  answer-processing (lowercase, punctuation/article stripping, number words). OK-VQA
  has multiple gold answers per question.
- **NExT-QA (T4):** multiple-choice; the program selects one of the candidate answers
  (via `select_answer` / logic). Report accuracy on Hard-Temporal, Hard-Causal, and
  the Full validation set.

Ambiguity in these protocols is the single most common cause of failed
reproductions; each is pinned to an open question below where not fully determined.

---

## 6. Modules and the pretrained models behind them

| API surface | Model (paper) | Variant / notes |
|-------------|---------------|-----------------|
| `find`, `exists` | **GLIP** | GLIP-L; detection threshold 0.5 |
| `verify_property` | **X-VLM** | threshold 0.6 |
| `best_image_match`, `best_text_match` | **X-VLM** (paper §4.1 also cites **CLIP** as an image-text similarity model) | config default = X-VLM (OQ-4) |
| `simple_query` | **BLIP-2** | `blip2-flan-t5-xxl`, 8-bit |
| `compute_depth` | **MiDaS** | DPT depth; returns median relative depth |
| `llm_query` (T3), `select_answer` (T4) | **GPT-3** `text-davinci-003` | external knowledge / MC selection |
| program generator `π` | **Codex** `code-davinci-002` | temperature 0, max_tokens 512 |

---

## 7. Hyperparameters (tagged by source)

| Parameter | Value | Source |
|-----------|-------|--------|
| Codex temperature | 0.0 | `[in official code]` (base_config.yaml) |
| Codex max_tokens | 512 | `[in official code]` |
| Codex best_of | 1 | `[in official code]` |
| GLIP detection threshold | 0.5 | `[in official code]` |
| `ratio_box_area_to_image_area` | 0.0 | `[in official code]` |
| `crop_larger_margin` | true (+10% crop) | `[in official code]` |
| X-VLM verify_property threshold | 0.6 | `[in official code]` |
| best-match model | X-VLM | `[in official code]` |
| BLIP-2 model | blip2-flan-t5-xxl, 8-bit | `[in official code]` |
| GPT-3 model / temp / n_votes | text-davinci-003 / 0.0 / 1 | `[in official code]` |
| batch_size | 20 | `[in official code]` |
| seeds | not stated | `[NOT SPECIFIED]` — assume single deterministic (temp 0) run; we add ≥3 where stochastic |

---

## 8. Authors' compute vs. ours

The method is **training-free**, so cost is inference only — but the modules are
heavy. GLIP-L and **BLIP-2 flan-t5-xxl** dominate GPU memory (BLIP-2 XXL ≈ 40 GB
fp16; ~20 GB in 8-bit). The paper does not tabulate GPU type/hours; the official
code runs models each in their own process with a producer–consumer batching
scheme, implying a multi-GPU or large-single-GPU setup.

**Feasibility on our target (university SLURM GPU cluster):** feasible. A single
modern 40–80 GB GPU (A100/H100) can host BLIP-2 8-bit + GLIP if scheduled
carefully; otherwise split models across GPUs via the multiprocessing path. The
open-weights code generator (replacing Codex) adds one more resident model — plan
in Phase 3 addresses memory scheduling and, if needed, running the code LLM on a
separate GPU or via a smaller quantized variant. Datasets are large (COCO images,
NExT-QA videos); disk quota is tracked in `docs/cluster_notes.md`.

---

## 9. Open questions → must resolve into `docs/deviations.md`

- **OQ-1 (RefCOCO metric): RESOLVED** — reported number is **mean IoU** (72.0/67.0);
  also compute secondary [email protected]. Watch the bottom-left `(left,lower,right,upper)`
  box convention. (See §5.)
- **OQ-2 (OK-VQA split):** Exact split used (val vs test) and VQA answer-processing
  variant. Confirm from official dataset code.
- **OQ-3 (video frame sampling):** NExT-QA frame stride / max frames per clip.
- **OQ-4 (best-match model):** Paper text says CLIP for best_image/text_match; config
  defaults to X-VLM. Decide and record which we use.
- **OQ-5 (program generator, MANDATORY deviation D1):** Codex is deprecated →
  open-weights code model. Which one, and its effect on program quality.
- **OQ-6 (llm_query / select_answer, deviation D2):** GPT-3 `text-davinci-003` is
  deprecated → open-weights instruct LLM. Which one.
- **OQ-7 (answer normalization):** exact GQA/OK-VQA answer-cleaning code path to
  match the official evaluator.
- **OQ-8 (GLIP build):** the official repo uses a *modified* GLIP with updated CUDA
  kernels for newer PyTorch; must build from their fork — a known integration risk.

Every item above resolves into a documented assumption or a `# DEVIATION:` entry
before the corresponding number is reported. Nothing is substituted silently.
