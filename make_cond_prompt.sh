#!/usr/bin/env bash
# Build prompts/api_depth_cond.prompt — condition C4.
#
# C3 (api_depth.prompt) measured:
#   RefCOCO+ depth queries   4.26% -> 19.15%   (4.5x, the target)
#   RefCOCO  spatial (2D)   34.26% -> 36.68%   (control holds)
#   RefCOCO  non-spatial    45.97% -> 36.02%   (-10.0, the cost)
#   RefCOCO+ non-spatial    33.55% -> 28.92%   (-4.6)
#
# depth_order is invoked 119 times on RefCOCO, where most queries need no depth at
# all. Given a new primitive the generator reaches for it indiscriminately. The
# hypothesis is that the loss is over-application, not a defect in the primitives.
#
# C4 changes ONE thing: it states when the depth functions apply and when they do
# not. Everything else — the API listing, the examples, the return-type contract —
# is byte-identical to C3, so any difference is attributable to the routing
# instruction alone.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path

src = Path("prompts/api_depth.prompt")
dst = Path("prompts/api_depth_cond.prompt")
s = src.read_text()

# Anchor on the guideline block introduced when api_depth.prompt was built.
old = ("- Resolve NEAR/FAR/BEHIND/IN FRONT using the depth functions below. "
       "These relations are about distance from the camera and cannot be "
       "recovered from image-plane coordinates.")
new = ("- Resolve NEAR/FAR/BEHIND/IN FRONT using the depth functions below. "
       "These relations are about distance from the camera and cannot be "
       "recovered from image-plane coordinates.\n"
       "- Use the depth functions ONLY when the query actually expresses a depth "
       "relation: closest, nearest, farthest, furthest, behind, in front of, "
       "between (in depth). For every other query they add nothing and cost "
       "accuracy.\n"
       "- If the query is about LEFT/RIGHT/TOP/BOTTOM, sort by patch coordinates "
       "and do not call any depth function.\n"
       "- If the query is about a colour, clothing, size or any other attribute, "
       "use verify_property or best_text_match and do not call any depth function.")
assert old in s, "guideline anchor not found — was api_depth.prompt built by make_depth_prompt.sh?"
s = s.replace(old, new)

# Two negative examples. The positive ones already exist; what is missing is a
# demonstration of when NOT to reach for depth.
counter = '''
Two queries that must NOT use the depth functions:

>>> # Query: the man on the left
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     man_patches = image_patch.find("man")
>>>     if not man_patches: return image_patch
>>>     man_patches.sort(key=lambda p: p.horizontal_center)
>>>     return man_patches[0]

>>> # Query: the woman in the red jacket
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     woman_patches = image_patch.find("woman")
>>>     if not woman_patches: return image_patch
>>>     red = [w for w in woman_patches if w.verify_property("woman", "red jacket")]
>>>     return red[0] if red else woman_patches[0]

'''

marker = "Query: INSERT_QUERY_HERE"
assert marker in s
s = s.replace(marker, counter.strip() + "\n\n" + marker)

dst.write_text(s)
print(f"  wrote {dst}")
print(f"  {len(src.read_text())} bytes (C3) -> {len(s)} bytes (C4)")
PYEOF

echo
echo "=== all prompt variants ==="
for f in prompts/api.prompt prompts/api_grounding.prompt \
         prompts/api_depth.prompt prompts/api_depth_cond.prompt; do
  printf "  %-34s %6s bytes  placeholder:%s\n" "$f" "$(wc -c < "$f")" \
    "$(grep -c INSERT_QUERY_HERE "$f")"
done
echo
echo "=== C4 routing instructions ==="
grep -n "ONLY when\|do not call any depth" prompts/api_depth_cond.prompt
