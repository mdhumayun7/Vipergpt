# Master comparison grid

Accuracy at IoU >= 0.5, greedy decoding, n = 500 per cell, testA split.
Every number is read from an archived run summary; nothing is transcribed.

Conditions: C1 released prompt, C2 + return-type contract,
C3 + depth primitives, C4 + primitives and routing rule.

## refcoco / testA

| Generator | C1 | C2 | C3 | C4 | C1→C2 | C3→C4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 0.80 | 34.40 | --- | --- | 33.6 | --- |
| Qwen2.5-Coder-32B | 0.00 | 41.60 | --- | --- | 41.6 | --- |
| Qwen2.5-Coder-7B | 0.00 | 39.20 | 36.40 | 42.00 | 39.2 | 5.6 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 29.60 | 38.80 | 41.80 | 29.6 | 3.0 |
| OpenCoder-8B | 0.20 | 40.80 | 37.40 | 43.20 | 40.6 | 5.8 |
| Yi-Coder-9B | 3.80 | 43.80 | 39.00 | 45.40 | 40.0 | 6.4 |

### refcoco — spatial subset

| Generator | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 0.69 | 34.95 | --- | --- |
| Qwen2.5-Coder-32B | 0.00 | 35.64 | --- | --- |
| Qwen2.5-Coder-7B | 0.00 | 34.26 | 36.68 | 37.02 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 28.37 | 33.91 | 35.29 |
| OpenCoder-8B | 0.35 | 35.64 | 38.06 | 41.18 |
| Yi-Coder-9B | 6.57 | 39.79 | 31.14 | 40.48 |

### refcoco — non-spatial subset

| Generator | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 0.95 | 33.65 | --- | --- |
| Qwen2.5-Coder-32B | 0.00 | 49.76 | --- | --- |
| Qwen2.5-Coder-7B | 0.00 | 45.97 | 36.02 | 48.82 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 31.28 | 45.50 | 50.71 |
| OpenCoder-8B | 0.00 | 47.87 | 36.49 | 45.97 |
| Yi-Coder-9B | 0.00 | 49.29 | 49.76 | 52.13 |

## refcoco+ / testA

| Generator | C1 | C2 | C3 | C4 | C1→C2 | C3→C4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-32B | --- | --- | --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 0.00 | 30.80 | 28.00 | 37.60 | 30.8 | 9.6 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 24.00 | 38.60 | 38.40 | 24.0 | -0.2 |
| OpenCoder-8B | 0.00 | 30.20 | 28.00 | 31.60 | 30.2 | 3.6 |
| Yi-Coder-9B | 0.80 | 34.40 | 35.60 | 43.80 | 33.6 | 8.2 |

### refcoco+ — spatial subset

| Generator | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | --- | --- | --- | --- |
| Qwen2.5-Coder-32B | --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 0.00 | 4.26 | 19.15 | 21.28 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 14.89 | 27.66 | 19.15 |
| OpenCoder-8B | 0.00 | 23.40 | 17.02 | 19.15 |
| Yi-Coder-9B | 0.00 | 25.53 | 29.79 | 19.15 |

### refcoco+ — non-spatial subset

| Generator | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | --- | --- | --- | --- |
| Qwen2.5-Coder-32B | --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 0.00 | 33.55 | 28.92 | 39.29 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 24.94 | 39.74 | 40.40 |
| OpenCoder-8B | 0.00 | 30.91 | 29.14 | 32.89 |
| Yi-Coder-9B | 0.88 | 35.32 | 36.20 | 46.36 |

---

## The four questions the grid answers

### Q1 — does the specification gap appear in every family? (C1 → C2)

| Generator | C1 | C2 | gain |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 0.80 | 34.40 | 33.6 |
| Qwen2.5-Coder-32B | 0.00 | 41.60 | 41.6 |
| Qwen2.5-Coder-7B | 0.00 | 39.20 | 39.2 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 0.00 | 29.60 | 29.6 |
| OpenCoder-8B | 0.20 | 40.80 | 40.6 |
| Yi-Coder-9B | 3.80 | 43.80 | 40.0 |

A gap present in every family cannot be a quirk of one model.

### Q2 — does the depth capability help in every family? (C2 → C3, depth subset)

| Generator | C2 spatial | C3 spatial | gain |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B | --- | --- | --- |
| Qwen2.5-Coder-32B | --- | --- | --- |
| Qwen2.5-Coder-7B | 4.26 | 19.15 | 14.9 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 14.89 | 27.66 | 12.8 |
| OpenCoder-8B | 23.40 | 17.02 | -6.4 |
| Yi-Coder-9B | 25.53 | 29.79 | 4.3 |

RefCOCO+ spatial is the depth-word subset (n = 47).

### Q3 — is the routing rule necessary in every family? (C3 → C4, non-spatial)

| Generator | C2 | C3 | C4 | C3 cost | C4 recovery |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B | 33.65 | --- | --- | --- | --- |
| Qwen2.5-Coder-32B | 49.76 | --- | --- | --- | --- |
| Qwen2.5-Coder-7B | 45.97 | 36.02 | 48.82 | -9.9 | 12.8 |
| DeepSeek-Coder-V2-Lite (16B MoE) | 31.28 | 45.50 | 50.71 | 14.2 | 5.2 |
| OpenCoder-8B | 47.87 | 36.49 | 45.97 | -11.4 | 9.5 |
| Yi-Coder-9B | 49.29 | 49.76 | 52.13 | 0.5 | 2.4 |

If C3 costs non-spatial accuracy in every family, over-application is a
property of the specification rather than of one generator.

### Q4 — does a stronger generator raise accuracy? (best condition)

| Generator | C4 RefCOCO | C4 RefCOCO+ | vs paper (72.0 / 67.0) |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B | --- | --- | --- |
| Qwen2.5-Coder-32B | --- | --- | --- |
| Qwen2.5-Coder-7B | 42.00 | 37.60 | 58% / 56% |
| DeepSeek-Coder-V2-Lite (16B MoE) | 41.80 | 38.40 | 58% / 57% |
| OpenCoder-8B | 43.20 | 31.60 | 60% / 47% |
| Yi-Coder-9B | 45.40 | 43.80 | 63% / 65% |

The perception ceiling measured on this stack is 81.5 (GLIP top-box, held-out
n = 200), which is above the paper's 72.0. Perception is therefore not the
limiting factor; program synthesis is.

