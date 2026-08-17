#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("docs/deviations.md")
s = p.read_text()

old = """**Defaults are OFF** (`max_detections=0`, `find_nms_iou=0`), so every reported number
predates and is unaffected by this change. Enabling it is a documented ablation, not
part of the main results."""

new = """**Defaults are OFF** (`max_detections=0`, `find_nms_iou=0`), so every reported number
predates and is unaffected by this change.

**MEASURED 2026-08-17 (jobs 29563, 29565), and the defaults now stay off for a
reason rather than by caution.** Capping to k candidates on the same C4 programs,
RefCOCO/testA:

| cap | overall | spatial | non-spatial |
|---|---:|---:|---:|
| off | 42.80 | 37.72 | 49.76 |
| k=12 | 45.20 | 41.52 | 50.24 |
| k=6 | 49.60 | 47.75 | 52.13 |
| k=3 | 57.60 | 61.59 | 52.13 |

57.60% would be the best number in this project by 15 points. A control disqualifies
it. Applying the identical cap to C2, which has no depth primitives at all:

| | no cap | k=3 | gain |
|---|---:|---:|---:|
| C2 spatial | 34.60 | 57.09 | +22.5 |
| C4 spatial | 37.72 | 61.59 | +23.9 |
| C2 non-spatial | 45.02 | 45.50 | +0.5 |
| C4 non-spatial | 49.76 | 52.13 | +2.4 |

C2 gains 22.5 points on spatial queries with no depth reasoning present. Only ~1.4 of
C4's 23.9 is attributable to depth. The mechanism is arithmetic: selecting among 3
candidates rather than 20 raises the hit rate by construction (random choice scores
33% at k=3 against 5% at k=20), and RefCOCO's referred object is nearly always among
the most confident detections. Non-spatial queries gain almost nothing, which
confirms it — they verify attributes rather than selecting among same-class
candidates.

**Effect on results:** none, deliberately. Enabling the cap would exploit a property
of the dataset rather than demonstrate a method. Recorded as a negative result.

**NMS is also left off.** It helps RefCOCO marginally (42.00 -> 42.80) but harms
RefCOCO+ depth queries badly (21.28 -> 12.77): suppressing overlapping boxes removes
the depth-distinct instances of the same object that a depth query exists to
distinguish.

**What survives:** C4's advantage over C2 holds under both settings (3.8 points
uncapped, 5.4 capped), so no claim in this work depends on the cap."""

assert old in s, "D12 anchor not found"
p.write_text(s.replace(old, new))
print("D12 updated with the control result")
PYEOF

grep -c "^## D" docs/deviations.md
wc -l docs/deviations.md
