# Deviations from the paper

Living document. Every substitution away from what the paper specifies is
recorded here, in a `# DEVIATION:` code comment at the site, and in the README
results table. No silent substitutions.

| # | Area | Paper specifies | This repo uses | Reason | Likely effect on metric |
|---|------|-----------------|----------------|--------|-------------------------|
| D1 | Program generator (π) | OpenAI Codex `code-davinci-002` | Open-weights code model (finalized in plan) | Codex API deprecated 2023-03-23; unavailable | Uncertain; open code models generate less reliable programs → likely small negative |
| D2 | External-knowledge LLM (`llm_query`) | GPT-3 `text-davinci-003` | Open-weights instruct LLM | Cost + offline-cluster requirement; text-davinci-003 also deprecated | Possible negative on OK-VQA (knowledge recall) |

More rows added as implementation proceeds.

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
