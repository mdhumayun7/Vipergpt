# Contribution spec — a depth-grounded spatial API for ViperGPT

**Status:** implemented and evaluated; see docs/research_log.md, 2026-08-15 to 08-17.
**Motivated by:** `results/reproduction_table.md`, Table 2, and
`results/spatial_audit.txt`.

---

## 1. The measured problem

Under the grounding contract, on identical infrastructure:

| Spatial type | Example | Resolvable by | Accuracy |
|---|---|---|---:|
| 2D image-plane | "man on right" | `sort(key=lambda p: p.horizontal_center)` | 35.29% (n=289) |
| Depth / relational | "man closest to us", "umpire behind catcher" | needs 3D geometry | **6.38%** (n=47) |

Three things make this a specification failure rather than a model failure:

1. **The API cannot express depth relations.** `compute_depth()` returns one scalar —
   the median relative depth of a patch. From that, "A is behind B" is expressible,
   but "the third-closest person", "between A and B in depth", "facing away", and
   metric size are not.
2. **The prompt actively discourages geometry.** It instructs the generator to
   "use base Python (comparison, sorting) for basic logical operations,
   left/right/up/down, math, etc."
3. **The generator complies.** Milestone 1 call frequency over 500 programs:
   `find` 547, `compute_depth` 29, `distance` 5.

So ViperGPT does not reason spatially; it sorts pixel coordinates, and where that is
insufficient it fails. The failure is silent — the program runs and returns a box.

---

## 2. Proposed API

Replace the scalar `compute_depth` with a metric back-projection and expose
operations over it. Each returns a plain Python value, so generated programs remain
readable and composable.

```python
class ImagePatch:
    def depth_map(self) -> np.ndarray:
        """Metric depth for this patch, in metres. (H, W)."""

    def position_3d(self) -> tuple[float, float, float]:
        """(X, Y, Z) of the patch centroid in camera coordinates, metres.
        Back-projected using estimated intrinsics; Z is robust-median depth
        over the patch interior, excluding the border to reject background."""

    def physical_size(self) -> tuple[float, float]:
        """(width, height) in metres, from the 3D extent of the patch."""

def distance_3d(a: ImagePatch, b: ImagePatch) -> float:
    """Euclidean distance in metres between patch centroids."""

def is_behind(a: ImagePatch, b: ImagePatch, margin: float = 0.3) -> bool:
    """True if a is at least `margin` metres further from the camera than b."""

def depth_order(patches: list[ImagePatch]) -> list[ImagePatch]:
    """Patches sorted near-to-far. Backs 'closest', 'nearest', 'farthest',
    'second closest', and 'third from the front' in one primitive."""

def is_between_3d(a: ImagePatch, b: ImagePatch, c: ImagePatch) -> bool:
    """True if a lies between b and c in 3D, by projection onto the b->c axis."""
```

`depth_order` is the highest-value single addition: the audit shows *closest*,
*nearest*, *farthest*, *closer* account for most of the RefCOCO+ spatial subset.

### Implementation

| Component | Choice | Note |
|---|---|---|
| Depth | Depth Anything V2 (metric) or ZoeDepth | Replaces MiDaS. MiDaS is relative-only; ordering works, metres do not. |
| Intrinsics | assumed FOV ≈ 60°, principal point at image centre | COCO has no intrinsics. **This is a deviation and must be logged.** Relative comparisons (ordering, betweenness) survive the assumption; absolute metres do not. |
| Back-projection | standard pinhole: `X = (u - cx) * Z / fx` | ~30 lines of numpy. |
| Patch depth | median over the central 50% of the patch | Border pixels are usually background and skew the median badly. |

Only the depth model changes; everything else is arithmetic. GLIP, CLIP, the
executor, and the evaluator are untouched.

---

## 3. Prompt changes

Two edits to `prompts/api_grounding.prompt`, producing a third variant:

1. **Remove the anti-geometry instruction.** Drop "left/right/up/down" from the
   base-Python guideline, keeping it for genuinely non-spatial logic.
2. **Add the new signatures with docstrings and two worked examples** — one depth
   ordering, one relational:

```python
# Query: the man closest to the camera
def execute_command(image) -> ImagePatch:
    image_patch = ImagePatch(image)
    men = image_patch.find("man")
    if not men: return image_patch
    return depth_order(men)[0]

# Query: the cup behind the bottle
def execute_command(image) -> ImagePatch:
    image_patch = ImagePatch(image)
    cups = image_patch.find("cup")
    bottles = image_patch.find("bottle")
    if not cups: return image_patch
    if not bottles: return cups[0]
    behind = [c for c in cups if is_behind(c, bottles[0])]
    return behind[0] if behind else cups[0]
```

---

## 4. Evaluation

The pipeline already exists. Three conditions, identical everything else:

| # | Condition | Purpose |
|---|---|---|
| C1 | released prompt | 0.00% floor (have it) |
| C2 | + grounding contract | 39.20% / 31.80%, spatial 35.29% / 6.38% (have it) |
| C3 | + depth-grounded API | **the contribution** |

**Primary claim:** C3 improves accuracy on the depth-query subset (RefCOCO+ spatial,
n=47) over C2's 6.38%.

**Necessary control:** C3 must not degrade the 2D subset (RefCOCO spatial, 35.29%)
or the non-spatial sets. A geometry-heavy prompt that hurts everything else has
traded one failure for another.

**Ablations, using the paper's own Section 5.2 interventional protocol:**

- Prompt-only: keep the scalar `compute_depth`, add only the geometric examples.
  Separates *encouraging* geometry from *providing* it.
- API-only: add the functions but keep the original guideline text.
- Neutralise `depth_order` (return the input order unchanged) and measure the drop.
  This is the direct test that the new primitive is load-bearing rather than
  decorative.

**Reporting:** all three conditions, three seeds, mean ± std. n=47 is small, so the
depth-subset result must be reported with its interval and never as a point estimate.
If the effect is real it should also appear on RefCOCO's own depth-word queries
(`behind`, `front of`, `closest`), which is an independent check on a different
sample.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Metric depth is unreliable without true intrinsics | Claim *ordering*, not metres. `depth_order` and `is_behind` need only monotonicity, which relative depth already provides. |
| n=47 is too small to carry the claim | Pool RefCOCO + RefCOCO+ depth-word queries; extend beyond testA if needed. |
| The generator ignores the new API, as it ignored `compute_depth` | Measure call frequency directly, as in Milestone 1. If `depth_order` is not invoked, the finding is about prompt adherence and is still reportable. |
| Improvement comes from the prompt, not the geometry | This is exactly what the prompt-only ablation isolates. |

---

## 6. Sequence

1. Swap MiDaS for a metric depth model; verify ordering against RefCOCO+
   depth queries by hand on ~10 images.
2. Implement `position_3d`, `depth_order`, `is_behind`, `distance_3d`. Unit-test
   each on synthetic geometry with known answers.
3. Write `prompts/api_depth.prompt`.
4. Generate (vipergpt env) → execute (glip_env), 3 seeds, both datasets.
5. Run the ablations.
6. Write up.

Steps 1–3 are the only new code. Steps 4–6 reuse the Milestone 2 pipeline unchanged.
