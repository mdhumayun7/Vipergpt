# Next session — start here

## 1. Unresolved bug (do this first)

test_score_filter.py variants C and D score 0/31 while their oracle is 45-48%.
Variants A/B on the same data score 22.6% / 9.7%. Verified NOT to be:
  - stale bytecode (cleared __pycache__, ran with python -B)
  - wrong module (inspect confirms the interpolate patch is loaded)
  - missing scores (GLIP returns 41 boxes, 41 scores, e.g. 0.611 0.452 0.448)
  - depth map aspect ratio (now resampled to image size, verified MATCH)
  - identical candidate sets (area-top and score-top differ)

Prime suspect: crop_region() in test_score_filter.py. It converts bottom-left box
coords to map rows via (H - upper) .. (H - lower). A manual check using
full[H-up : H-lo, l : r] produced sensible, varied depths (1288, 1501, 1271, 1668),
so compare the two readers side by side on ONE image and diff the values.

Second suspect: numbers were byte-identical across three runs with different code in
between, which is suspicious in itself. Confirm the test is not short-circuiting
(e.g. `continue` skipping every C/D case, leaving hit=0 with n incremented elsewhere).

## 2. Then: promote the prototype to API

- ImagePatch: depth_order(patches), is_behind(a,b), distance_3d(a,b), position_3d()
- ModuleBus: register them
- Carry variant A config: per-crop depth, area top-6, GLIP thr 0.2
- Unit tests on synthetic geometry with known answers

## 3. Then: prompts/api_depth.prompt

- Remove "use base Python for left/right/up/down"
- Add signatures + two worked examples (depth ordering, behind-relation)

## 4. Then: three-condition run

  C1 released prompt        0.00%   (have)
  C2 grounding contract    39.20% / 6.38% on depth subset  (have)
  C3 depth API             ?

Plus ablations: neutralise depth_order (must cause a drop), and prompt-only
(separates encouraging geometry from providing it). Measure call frequency directly
— if depth_order is not invoked, the finding is about prompt adherence.

## Still open from Milestone 2

- X-VLM proper (D9); CLIP substitute in use
- Final table: all conditions x 3 seeds

## Reference numbers

  depth signal, GT boxes only : 80% correct vs 26% chance (n=25)
  filtering: 39.5 boxes -> 9.7% | 5.8 boxes -> 22.6% | oracle 61-68%
  scale sweep spatial         : 34.95 / 35.29 / 34.95 (1.5B / 7B / 32B) — flat
  scale sweep non-spatial     : 33.65 / 44.55 / 48.82 — rises
