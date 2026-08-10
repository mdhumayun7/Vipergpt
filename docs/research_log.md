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
