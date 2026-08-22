# Project Handover — ViperGPT Reproduction and Extension

**Paste this as the first message in a new chat.** It contains everything needed to
continue without re-explaining anything.

---

## 0. Who and what

**Student:** MD Humayun
**Programme:** M.Tech, 2nd year, CSE — specialisation Information Security and Privacy
**Institute:** SVNIT Surat
**Supervisor:** Dr. M. A. Zaveri
**Repository:** `https://github.com/mdhumayun7/Vipergpt` (private), branch `main`

**Working style — follow this:**
- Discussion and process in **Hinglish**; all dissertation/report content in **formal
  academic English**.
- Never fabricate a number. If a figure is not in the archived summaries, say so.
- Every claim must trace to a run summary, a record file, or a SLURM job ID.
- Negative results are reported, not hidden. Two have already been reported this way.
- Prefer a small verification before a long job. Dry runs have already saved four
  wasted GPU-hours twice.
- The user often cannot upload files successfully — **uploaded documents frequently
  arrive blank**. Ask for terminal output pasted directly into chat, or screenshots.
  Screenshots work reliably; file attachments usually do not.

---

## 1. The paper being reproduced

**Surís, Menon & Vondrick, "ViperGPT: Visual Inference via Python Execution for
Reasoning", ICCV 2023** (arXiv:2303.08128v1).

**Method.** A code-generating LLM writes a Python program over a vision API; the
program is executed against the image.

    z = π(q)            program generation from the query
    r = φ(x, z)         execution against the image

The generator sees only the query and an API specification prompt — never the image.
Training-free: no gradient step anywhere.

**Modules in the paper:** GLIP (`find`), X-VLM (`verify_property`,
`best_text_match`), BLIP-2 (`simple_query`), MiDaS (`compute_depth`), GPT-3
(`llm_query`), Codex `code-davinci-002` as the generator.

**Reported (Table 1, testA):** RefCOCO **72.0**, RefCOCO+ **67.0**.
Zero-shot baselines it reports alongside: ReCLIP 58.6/60.5, GLIP 55.0/52.2,
OWL-ViT 30.3/29.4.

**Why reproduction is non-trivial:** Codex was withdrawn. Any present-day
reproduction must substitute a generator, and that substitution is what the whole
dissertation turns on.

**Metric ambiguity in the paper, already handled:** the Table 1 caption says
"accuracy" while its column header says "IoU (%)", and the released evaluator
computes mean IoU. This work reports **accuracy at IoU ≥ 0.5** as primary and mean
IoU alongside; on these runs the two agree to about one point. The released
evaluator also substitutes the dataset mean box (≈22.6 IoU) for a missing
prediction; this work scores an invalid return as **0**. Both facts are stated
explicitly in the dissertation.

---

## 2. What the dissertation actually claims

Four milestones, each a hypothesis that was tested. Two hypotheses were rejected.

**M1 — the specification gap.** The released prompt never states that a grounding
program must return an `ImagePatch`, and five of its seven worked examples declare
`str`. With an open-weights generator, 98.8% of programs return a string where a box
is required; executed accuracy is **0.00** — not a crash, since 332 of 500 programs
run to completion. Adding a three-sentence return-type contract gives **39.20**.

**M2 — the spatial deficit is an interface problem.** Across 1.5B / 7B / 32B,
non-spatial accuracy rises 33.65 → 48.82 while spatial accuracy stays flat at
34.95 / 34.26 / 34.95. A larger generator writes better programs for everything
except spatial relations. The API exposes one scalar for depth
(`compute_depth`), from which inter-object ordering cannot be expressed, and the
prompt directs the generator away from geometry.

**M3 — a depth-grounded API, and the routing finding.** Added `depth_order`,
`is_behind`, `is_in_front_of`, `is_between_3d`, `distance_3d`. Offered
unconditionally (C3) they raise depth queries 4.26 → 19.15 but cost ten points of
non-spatial accuracy — the generator over-applies a new primitive. Adding a rule
stating *when* they apply (C4) improves **every** subset: depth queries to 21.28,
RefCOCO overall to 42.00.

**M4 — cross-family generalisation.** Four generator families. The specification gap
appears in all of them; Yi-Coder-9B is the strongest generator measured.

**The two results share a form:** in C1 the specification failed to say *what* to
return; in C3 it failed to say *when* a capability applies. Neither is a model
limitation.

---

## 3. The four prompt conditions — the experimental variable

Each is a strict superset of the previous. Model, modules, data and decoding are held
fixed throughout.

| | File | Bytes | What it adds |
|---|---|---:|---|
| **C1** | `prompts/api.prompt` | 7,645 | nothing — the released specification |
| **C2** | `prompts/api_grounding.prompt` | 8,639 | explicit return-type contract + 2 grounding examples |
| **C3** | `prompts/api_depth.prompt` | 11,485 | 5 depth primitives + 3 examples; removes the "use base Python for left/right/up/down" instruction |
| **C4** | `prompts/api_depth_cond.prompt` | 12,680 | 3 routing rules stating *when* depth applies + 2 negative examples |

C2 text added:
```
This is a VISUAL GROUNDING task. The function must return the single
ImagePatch that the query refers to. It must NOT return a string, a bool,
or a list.
Resolve every spatial relation in the query; do not drop it.
```

C4 text added:
```
- Use the depth functions ONLY when the query actually expresses a depth
  relation: closest, nearest, farthest, furthest, behind, in front of,
  between (in depth). For every other query they add nothing and cost accuracy.
- If the query is about LEFT/RIGHT/TOP/BOTTOM, sort by patch coordinates and
  do not call any depth function.
- If the query is about a colour, clothing, size or any other attribute, use
  verify_property or best_text_match and do not call any depth function.
```

**Key evidence from the released prompt itself:** its only example that returns an
`ImagePatch` sits inside `compute_depth`'s docstring, and it sorts ascending then
takes `[-1]` for "furthest away" — so the specification requires that a *larger*
`compute_depth` mean *further*. That independently justifies the sign correction in
D11.

---

## 4. Headline results (all archived; nothing here is typed from memory)

Accuracy at IoU ≥ 0.5, Qwen2.5-Coder-7B, greedy, n=500 per cell, testA.

| Dataset | Subset | C1 | C2 | C3 | C4 |
|---|---|---:|---:|---:|---:|
| RefCOCO | overall | 0.00 | 39.20 | 36.40 | **42.00** |
| RefCOCO | spatial (n=289) | 0.00 | 34.26 | 36.68 | 37.02 |
| RefCOCO | non-spatial (n=211) | 0.00 | 45.97 | 36.02 | 48.82 |
| RefCOCO+ | overall | 0.00 | 30.80 | 28.00 | **37.60** |
| RefCOCO+ | spatial (n=47) | 0.00 | 4.26 | 19.15 | **21.28** |
| RefCOCO+ | non-spatial (n=453) | 0.00 | 33.55 | 28.92 | 39.29 |

Mean IoU, RefCOCO C1–C4: 0.0000 / 0.3875 / 0.3655 / 0.4181.

**Scale sweep (C2, RefCOCO):** 1.5B 34.40, 7B 39.20, 32B 41.60.
Task-invalid under C1: 95.8% / 98.8% / 87.6%.
Spatial accuracy: 34.95 / 34.26 / 34.95 — flat.

**Seed variance (T=0.7, 3 seeds, RefCOCO):**

| | overall | spatial | non-spatial |
|---|---|---|---|
| C2 | 36.07 ± 1.75 | 32.87 ± 3.35 | 40.44 ± 0.72 |
| C3 | 34.40 ± 0.40 | 34.02 ± 0.80 | 34.92 ± 0.55 |
| C4 | **42.07 ± 0.61** | 38.29 ± 0.72 | 47.23 ± 1.19 |

Spatial variance falls 4.6× from C2 to C4 — the primitives buy *stability*, not only
accuracy. C2's greedy 39.20 sits 1.8 s.d. above its own sampled mean, so the greedy
table understates the C4 advantage; the honest separation is 6.0 points.

**Seed variance (RefCOCO+):** C4 overall 37.33 ± 0.42, spatial **19.15 ± 2.13**,
non-spatial 39.22 ± 0.56. At n=47 one query is 2.13 points, and the three seeds score
8/47, 9/47, 10/47 — **the standard deviation is exactly ±1 sample**. Greedy's 21.28 is
the top of the seed range, not the centre; both must be quoted.

**Four-model grid (C4, greedy, n=500):**

| Generator | C1 RefCOCO | C4 RefCOCO | C4 RefCOCO+ | share of 72.0 / 67.0 |
|---|---:|---:|---:|---|
| Qwen2.5-Coder-7B | 0.00 | 42.00 | 37.60 | 58% / 56% |
| DeepSeek-Coder-V2-Lite | 0.00 | 41.80 | 38.40 | 58% / 57% |
| OpenCoder-8B | 0.20 | 43.20 | 31.60 | 60% / 47% |
| **Yi-Coder-9B** | 3.80 | **45.40** | **43.80** | **63% / 65%** |

**Perception ceiling: 81.5%** (GLIP alone, top box, held-out n=200, threshold 0.2).
This is *above* the paper's 72.0, so perception is not the limiting factor — program
synthesis is.

**API adoption (RefCOCO, 500 programs):**

| Condition | `find` | `compute_depth` | `depth_order` | other prims |
|---|---:|---:|---:|---:|
| C1 | 548 | 29 | 0 | 0 |
| C2 | 639 | 23 | 0 | 0 |
| C3 | 598 | 19 | 119 | 63 |
| C4 | 559 | — | 37 | 7 |

---

## 5. The two negative results — reported, not hidden

**The candidate cap.** Capping `find` to 3 candidates raises C4 on RefCOCO from
42.80 to **57.60** — the largest number in the project. A control disqualified it:
applying the identical cap to **C2**, which contains no depth primitives, raises its
spatial accuracy by **+22.5** points against C4's **+23.9**. Only ~1.4 points are
attributable to depth. Mechanism: choosing among 3 rather than 20 raises the hit rate
by construction (random alone scores 33% at k=3 against 5% at k=20), and non-spatial
queries gain almost nothing (+0.5), which confirms it. `max_detections` stays
**disabled** in every reported number. NMS is also off: it helps RefCOCO marginally
(42.00 → 42.80) but harms RefCOCO+ depth queries badly (21.28 → 12.77).

**The depth sign.** MiDaS predicts *inverse* depth. A synthetic test scene said no
inversion was needed and cost hours — it was a brightness gradient the model read as
depth. Settled on ground-truth boxes with no detector: **80%** correct with the
inversion, **8%** without, against **26%** chance. Below chance means inverted, not
weak. Now pinned by 18 unit assertions in `test_depth_api.py`.

**Lesson recorded:** synthetic scenes are unreliable for validating perceptual
signals; validate against real annotations, and do it first.

---

## 6. Cluster and environment

**Machine:** SVNIT HPC, host `svnithpc`, nodes `node1` and `node2`, one NVIDIA
**H100 NVL (95 GB)** each. SLURM with **fractional GPU shards** (94 shards ≈ 94 GB).

**Hard constraints:**
- **Two queued jobs per user** (`AssocMaxSubmitJobLimit`). A third submission is
  rejected regardless of free GPU.
- **Compute nodes have no internet.** All weights and data must be fetched on the
  login node first. Jobs set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- **`git` is not installed on compute nodes** — hence `git_sha: nogit` in every
  execution summary. Provenance rests on SLURM job IDs and committed job scripts.
- **Quota 500 GB, currently ~470 GB used, ~30 GB free.** Watch this.
- `conda activate` fails silently inside SLURM batch shells — job scripts must
  prepend the env `bin` to `PATH` explicitly and set `PYTHONNOUSERSITE=1`.

**Two conda environments, deliberately separate** (they cannot coexist in one
interpreter — the GLIP fork needs an older torch than the generator):

| Env | Stage | torch | notes |
|---|---|---|---|
| `vipergpt` | generation | 2.13.0 | transformers 5.x, plus `tiktoken`, `blobfile`, `sentencepiece`, `protobuf` |
| `glip_env` | execution | 2.1.2+cu121 | transformers 4.36.2, numpy < 2 |

Activation pattern used in every job script:
```bash
module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }
```

**That last assertion is mandatory.** An early 500-sample run executed on a login
node with no GPU: GLIP raised `RuntimeError` 481 times, the run finished in 7
seconds, and produced a complete and entirely fake 0.00/0.00 table. Silent CPU
fallback is the most dangerous failure mode in this project because it yields
plausible output rather than an error.

**Typical shard requests:** GLIP only 12 · execution (GLIP+CLIP+MiDaS) 20–25 ·
Qwen-7B gen 25 · Yi-9B 28 · OpenCoder-8B 26 · DeepSeek-V2-Lite 40–45 · Qwen-32B 70+.

**Interactive session:**
```bash
srun --partition=gpu --gres=shard:25 --cpus-per-task=8 --mem=64G --time=00:40:00 --pty bash
```

---

## 7. Directory structure

```
/home/mazaveri/hpc-prog/humayun/
├── vipergpt/                      ← the repository, git root
│   ├── src/vipergpt_repro/
│   │   ├── models/
│   │   │   ├── glip.py            GLIP wrapper. tensor_inputs=True fix; exposes
│   │   │   │                      last_scores; max_detections + find_nms_iou knobs
│   │   │   │                      (both DISABLED by default — see D12)
│   │   │   ├── depth.py           MiDaS via cached HF Intel/dpt-hybrid-midas.
│   │   │   │                      Sign inverted (larger = further), map resampled
│   │   │   │                      to image size, central-50% statistic. D11.
│   │   │   ├── clip_vlm.py        CLIP ViT-L/14 — substitutes X-VLM. D9.
│   │   │   ├── _xvlm_backbone.py  stub, raises NotImplementedError
│   │   │   └── blip2.py           present, never loaded (D2/D10)
│   │   ├── pipeline/
│   │   │   ├── image_patch.py     ImagePatch + the contributed primitives:
│   │   │   │                      depth_order, is_behind, is_in_front_of,
│   │   │   │                      is_between_3d, distance_3d
│   │   │   ├── executor.py        restricted namespace, SIGALRM timeout,
│   │   │   │                      errors returned as data
│   │   │   ├── vision_bus.py      ModuleBus; D10 empty-string fallbacks
│   │   │   └── video_segment.py   present, unused
│   │   ├── eval/
│   │   │   ├── codegen_analysis.py     generation + static analysis.
│   │   │   │                          --model --prompt --temperature --seed
│   │   │   │                          --batch-size --max-samples --version --split
│   │   │   │                          trust_remote_code is conditional: only
│   │   │   │                          ("OpenCoder" in a.model) — see D15
│   │   │   ├── execute_refcoco.py      execution + scoring. --timeout.
│   │   │   │                          Reads MAX_DET / FIND_NMS from env (D12).
│   │   │   ├── return_type_analysis.py task-invalid + spatial-drop counter
│   │   │   └── metrics.py
│   │   ├── data/                  refcoco.py used; gqa/okvqa/nextqa scaffolded,
│   │   │                          never run (D2)
│   │   └── utils/                 paths, logging, seed, run
│   ├── prompts/
│   │   ├── api.prompt                  C1  (released, unmodified)
│   │   ├── api_grounding.prompt        C2
│   │   ├── api_depth.prompt            C3
│   │   └── api_depth_cond.prompt       C4
│   ├── configs/default.yaml       ⚠ describes NO run actually performed — the eval
│   │                               scripts build config inline. Documented as D14.
│   ├── docs/
│   │   ├── research_log.md        ~1150 lines, the full arc including failures
│   │   ├── deviations.md          D1–D18, each with reason and measured effect
│   │   ├── contribution_spec.md
│   │   ├── paper_spec.md          what the paper specifies, incl. OQ resolutions
│   │   ├── dissertation.tex       ~50 pages, 12 chapters, 16 figures
│   │   └── viper_c1c4_notes.tex   10-page professor-facing C1–C4 explanation
│   ├── results/
│   │   ├── m1_summaries/          12 static-analysis JSONs
│   │   ├── m2_summaries/          15
│   │   ├── m3_summaries/          29+
│   │   ├── grid_summaries/        36  (the four-model grid)
│   │   ├── records/               170+ per-sample JSONL (iou, kind, error, boxes)
│   │   ├── programs/              50+ generated-program JSONL — static evidence
│   │   ├── figures/               fig1–fig18, PDF + PNG
│   │   ├── reproduction_table.md  regenerate: python make_repro_table.py
│   │   ├── grid_master.md / .tex  regenerate: python make_grid_tables.py
│   │   ├── threshold_sweep.json
│   │   ├── spatial_audit.txt      manual audit of all 47 RefCOCO+ spatial queries
│   │   └── MANIFEST.md
│   ├── outputs/runs/              ← GITIGNORED. Timestamped run dirs.
│   │   │                            Naming: <ts>__<sha>__m1_<dataset>_testA_
│   │   │                                     <model>_<condition>_greedy
│   │   │                            Each holds programs.jsonl / records.jsonl /
│   │   │                            summary.json / analysis.md
│   ├── outputs/slurm/             job logs
│   ├── tests/                     20 unit tests + test_depth_api.py (18 assertions)
│   ├── make_figures.py            fig1–fig13
│   ├── make_arch_figures.py       fig14 architecture, fig15 paper-vs-ours
│   ├── make_comparison_table.py   fig16 + comparison_table.{md,tex}
│   ├── make_grid_tables.py        fig17, fig18, grid_master.{md,tex}
│   ├── make_repro_table.py
│   ├── run_grid_gen.sh <key>      keys: qwen7b qwen1p5b qwen32b deepseek
│   │                                    yicoder opencoder
│   ├── run_grid_exec.sh           auto-discovers and scores unscored cells
│   └── (many run_*.sh job scripts)
│
├── vipergpt_store/                ← data + weights, gitignored
│   ├── data/refcoco/{refcoco,refcoco+}/    refs(unc).p + instances.json
│   ├── data/coco/train2014/                82,783 images
│   └── pretrained_models/GLIP/
│       ├── checkpoints/glip_large_model.pth   (6.5 GB)
│       └── configs/glip_Swin_L.yaml
│
└── viper_official/                ← reference clone of cvlab-columbia/viper (80 MB)
                                     GLIP built from here; 7 .cu files patched
                                     (THC headers → ATen), 5 .py files patched
                                     (NumPy 2.0 aliases), sm_80;sm_90 targets
```

**HuggingFace cache:** `/home/mazaveri/hf_cache` — Qwen-7B (15G), Qwen-1.5B (2.9G),
Yi-Coder-9B (20G), DeepSeek-V2-Lite (31G), OpenCoder-8B (16G), CLIP ViT-L (6.4G),
MiDaS dpt-hybrid (468M), bert-base-uncased (421M). **Qwen-32B was deleted** to free
quota — its results are archived, redownload if needed.

---

## 8. Deviations D1–D18 (one line each; full text in `docs/deviations.md`)

| ID | What |
|---|---|
| D1 | Codex → Qwen2.5-Coder (and later 3 more families). Codex retired. Dominant term in the gap. |
| D2 | Scope: RefCOCO/RefCOCO+ only. GQA/NExT-QA storage; OK-VQA needs GPT-3, also retired. |
| D3 | RefCOCO annotations from Internet Archive — the UNC host no longer resolves. |
| D4 | Dependency versions differ from the pinned environment. |
| D5 | GLIP rebuilt for sm_90. CUDA 11.6 has no Hopper support, so the published env cannot run at all. |
| D6 | NumPy 2.0 aliases fixed in 5 GLIP files. |
| D7 | GLIP fetches a BERT tokenizer at load; cached in advance, offline asserted. |
| D8 | Detector threshold 0.5 → 0.2. Re-swept held out (n=200): 0.2 is within 1.0 point of optimal. |
| D9 | X-VLM → CLIP ViT-L. X-VLM weights are Drive-only. Costliest on RefCOCO+. |
| D10 | `simple_query` / `llm_query` return `""`. Favours C1; no effect on later conditions. |
| D11 | Depth backend: offline load, sign inverted, map resampled. Evidence 80% vs 8% vs 26% chance. |
| D12 | GLIP scores exposed; `max_detections` + NMS knobs added, **both off by default**. |
| D13 | `crop_larger_margin` disabled (paper sets true). Not deliberate; effect unmeasured. |
| D14 | `configs/default.yaml` describes no run actually performed — eval builds config inline. |
| D15 | `trust_remote_code` **only** for OpenCoder. Enabling globally broke DeepSeek. |
| D16 | Added tiktoken/blobfile (Yi) and sentencepiece/protobuf (OpenCoder). |
| D17 | DeepSeek batch size reduced to 1 — batch 6 gave parse=0.4% corruption, not a crash. |
| D18 | Grid and scale sweep share only the Qwen-7B cell; not fully crossed. |

---

## 9. Where things stand right now

**Complete:** M1, M2, M3, M4. Seeds for RefCOCO (C2/C3/C4) and RefCOCO+ (C3/C4).
Threshold validated held-out. Two controls run and reported. 18 figures. 50-page
dissertation compiling clean, 12 chapters, Milestone 4 is Chapter 9. A 10-page
professor-facing C1–C4 note. Everything pushed to GitHub.

**Not done, deliberately:**
- X-VLM (D9) — still substituted by CLIP
- Qwen-1.5B / 32B not extended to the full grid (12 cells missing)
- Grid cells are single-seed
- `crop_larger_margin` effect unmeasured

**Supervisor's instruction (received, partially acted on):** *"Qwen didn't beat the
paper — try 3–4 other LLMs and compare."* Done: four families now. Answer is that
Yi-Coder-9B is best at 45.40/43.80 (63%/65% of the paper) and that substituting the
generator buys 3–6 points, not 30.

---

## 10. What the user wants to do next

**Stated goal: beat or approach the paper's 72.0.**

The measured facts that constrain this:
- perception ceiling is **81.5**, so 72.0 is not impossible in principle
- best current is **45.40**, so ~36 points are lost between detection and selection
- Figure 7 shows C4's residual failure is almost entirely *"box returned, IoU < 0.5"*
  — the program returns a box and picks the **wrong object**. Errors are ~2%.
- so the bottleneck is **candidate selection**, not program generation

**Ranked plan agreed with the user:**

1. **C5 — "descriptive find"** (1 day, highest expected value).
   Generated programs currently pass **bare nouns** to `find()` and then filter with
   CLIP:
   ```python
   people = image_patch.find("person")
   red = [p for p in people if p.verify_property("person", "red jacket")]
   ```
   But GLIP is a *grounded* detector and accepts full phrases:
   ```python
   red = image_patch.find("person in a red jacket")
   ```
   C5 is a fifth prompt condition instructing the generator to pass the descriptive
   phrase to `find` when the query contains attributes. It targets RefCOCO+ directly,
   where accuracy is weakest and where D9 (CLIP-for-X-VLM) hurts most. It is the
   natural next step in the same story: C2 said *what to return*, C4 said *when a
   capability applies*, C5 says *how much of the query to give the detector*.

   **Diagnostic to run first** — confirms bare nouns are being passed:
   ```bash
   python -c "
   import json, glob, re, collections
   d = sorted(glob.glob('outputs/runs/*m1_refcoco+_testA_Yi-Coder-9B-Chat_depth_cond_greedy'))[-1]
   r = [json.loads(l) for l in open(d+'/programs.jsonl')]
   args = []
   for x in r: args += re.findall(r'find\(\s*[\"\\']([^\"\\']+)', x['program'])
   w = [len(a.split()) for a in args]
   print('calls:', len(args), 'mean words:', round(sum(w)/len(w),2))
   print(collections.Counter(w).most_common()); print(args[:12])
   "
   ```
   Mean ≈ 1.0 confirms the hypothesis.

2. **X-VLM (D9)** — 2–3 days. Closes the substitution most likely to be costing
   accuracy on RefCOCO+.

3. **Seeds for the grid** — currently single-run.

4. **Self-consistency** (generate k=5 programs, majority-vote the box) — cheap, but
   deviates from the paper's temperature-0 protocol, so must be a separate condition.

**Two ideas the user raised, with the assessment already given:**

- *"Try an even bigger model."* Scale evidence says 1.5B→32B (20×) buys +7.2 points.
  Extrapolation suggests a 70B model reaches ~50, not 72. Also 70B in bf16 is 140 GB
  against ~30 GB of free quota. Better to spend the effort on C5/X-VLM with Yi-Coder.

- *"Fine-tune the best model."* **This breaks the comparison.** ViperGPT's central
  claim is that it is training-free; a fine-tuned generator cannot be compared to a
  zero-shot 72.0, and Codex was not fine-tuned either. It is defensible only as a
  clearly separate condition: LoRA on Yi-Coder using (query, program) pairs written
  on the RefCOCO **train** split — never testA — reported in its own table as
  *"zero-shot 45.40 → supervised X"*, never against 72.0. Framed that way it answers
  a new question (how much does a little supervision buy?) rather than contaminating
  the reproduction. Note also that **81.5 is still the ceiling**: no amount of
  fine-tuning passes it without changing the detector, and changing the detector
  makes it a different paper.

---

## 11. Commands worth knowing

```bash
cd ~/hpc-prog/humayun/vipergpt

# state
git status --short && git log --oneline -3
quota -s | tail -1
squeue -u $USER

# regenerate every table and figure from archived summaries
python make_repro_table.py
python make_grid_tables.py
python make_figures.py --qual
python make_arch_figures.py
python make_comparison_table.py

# generation (vipergpt env) — one model, all 4 conditions, both datasets, resumable
sbatch --gres=shard:28 run_grid_gen.sh yicoder

# execution (glip_env) — auto-discovers everything unscored
sbatch run_grid_exec.sh

# archive after any run (outputs/ is gitignored)
for d in outputs/runs/*<tag>*; do
  [ -f "$d/summary.json" ]  && cp "$d/summary.json"  "results/grid_summaries/$(basename $d).json"
  [ -f "$d/records.jsonl" ] && cp "$d/records.jsonl" "results/records/$(basename $d).jsonl"
  [ -f "$d/programs.jsonl" ]&& cp "$d/programs.jsonl" "results/programs/$(basename $d).jsonl"
done
```

**Traps already hit — do not repeat:**
- A run directory is created *before* the model loads, so a crash leaves an empty
  directory. Resume logic must test for `programs.jsonl`, not the directory.
- `.gitattributes` had `*.png filter=lfs`, and `git-lfs` is not installed on the
  cluster — pushes failed until the LFS lines were removed.
- Killing an `hf download` leaves stale locks in `~/hf_cache/hub/.locks/`; the next
  attempt waits forever. Remove the lock directory.
- HF Xet transfer sometimes fails with a CAS error; retry with
  `HF_HUB_DISABLE_XET=1`.
- A truncated `tokenizer.model` produces a parse error, not a checksum failure.
  Verify tokenizers as well as weight shards.
