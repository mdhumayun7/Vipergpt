# Deviations from the paper

Living document. Every substitution away from what the paper specifies is
recorded here, in a `# DEVIATION:` code comment at the site, and in the README
results table. No silent substitutions.

| # | Area | Paper specifies | This repo uses | Reason | Likely effect on metric |
|---|------|-----------------|----------------|--------|-------------------------|
| D1 | Program generator (π) | OpenAI Codex `code-davinci-002` | Open-weights code model (finalized in plan) | Codex API deprecated 2023-03-23; unavailable | Uncertain; open code models generate less reliable programs → likely small negative |
| D2 | External-knowledge LLM (`llm_query`) | GPT-3 `text-davinci-003` | Open-weights instruct LLM | Cost + offline-cluster requirement; text-davinci-003 also deprecated | Possible negative on OK-VQA (knowledge recall) |

More rows added as implementation proceeds.
