#!/usr/bin/env bash
# Build prompts/api_depth.prompt — the third and final prompt condition.
#
# Two changes from api_grounding.prompt, deliberately kept separable so a
# prompt-only ablation can distinguish "encouraging geometry" from "providing it":
#
#   1. Remove the instruction to use base Python for left/right/up/down. That line is
#      why compute_depth is called 29 times against 547 for find: the prompt tells the
#      generator not to reach for geometry.
#   2. Document the depth primitives and show two worked examples.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path

src = Path("prompts/api_grounding.prompt")
dst = Path("prompts/api_depth.prompt")
s = src.read_text()

# ---- 1. drop the anti-geometry instruction, keep base Python for real logic
old_guide = ("- Use base Python (comparison, sorting) for basic logical operations, "
             "left/right/up/down, math, etc.")
new_guide = ("- Use base Python (comparison, sorting) for logical operations and arithmetic.\n"
             "- Resolve LEFT/RIGHT/ABOVE/BELOW using patch coordinates "
             "(left, right, horizontal_center, vertical_center).\n"
             "- Resolve NEAR/FAR/BEHIND/IN FRONT using the depth functions below. "
             "These relations are about distance from the camera and cannot be "
             "recovered from image-plane coordinates.")
if old_guide in s:
    s = s.replace(old_guide, new_guide)
    print("  removed the anti-geometry instruction")
else:
    print("  WARNING: guideline line not found; prompt may already be modified")

# ---- 2. document the primitives and show them in use
depth_api = '''
The following functions are also available. compute_depth() returns a distance from
the camera: a LARGER value means the object is FURTHER AWAY.

def depth_order(patches: List[ImagePatch], reverse: bool = False) -> List[ImagePatch]:
    """Returns the patches sorted from NEAREST to FURTHEST from the camera.
    Pass reverse=True to sort furthest to nearest.
    Examples
    --------
    >>> # Query: the closest dog
    >>> def execute_command(image) -> ImagePatch:
    >>>     image_patch = ImagePatch(image)
    >>>     dog_patches = image_patch.find("dog")
    >>>     if not dog_patches: return image_patch
    >>>     return depth_order(dog_patches)[0]
    >>> # Query: the second closest dog
    >>> #     return depth_order(dog_patches)[1]
    >>> # Query: the farthest dog
    >>> #     return depth_order(dog_patches)[-1]
    """

def is_behind(a: ImagePatch, b: ImagePatch, margin: float = 0.0) -> bool:
    """Returns True if a is further from the camera than b."""

def is_in_front_of(a: ImagePatch, b: ImagePatch, margin: float = 0.0) -> bool:
    """Returns True if a is closer to the camera than b."""

def is_between_3d(a: ImagePatch, b: ImagePatch, c: ImagePatch) -> bool:
    """Returns True if a lies between b and c in depth."""

def distance_3d(a: ImagePatch, b: ImagePatch) -> float:
    """Returns the separation between two patches, combining image-plane offset and
    depth. Relative, not metres — use it to compare, not to measure."""

More examples:

>>> # Query: the man closest to the camera
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     man_patches = image_patch.find("man")
>>>     if not man_patches: return image_patch
>>>     return depth_order(man_patches)[0]

>>> # Query: the cup behind the bottle
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     cup_patches = image_patch.find("cup")
>>>     bottle_patches = image_patch.find("bottle")
>>>     if not cup_patches: return image_patch
>>>     if not bottle_patches: return cup_patches[0]
>>>     behind = [c for c in cup_patches if is_behind(c, bottle_patches[0])]
>>>     return behind[0] if behind else cup_patches[0]

>>> # Query: the person on the left who is nearest to us
>>> def execute_command(image) -> ImagePatch:
>>>     image_patch = ImagePatch(image)
>>>     people = image_patch.find("person")
>>>     if not people: return image_patch
>>>     people.sort(key=lambda p: p.horizontal_center)
>>>     left_half = people[: max(1, len(people) // 2)]
>>>     return depth_order(left_half)[0]

'''

marker = "Query: INSERT_QUERY_HERE"
assert marker in s, "INSERT_QUERY_HERE not found"
s = s.replace(marker, depth_api.strip() + "\n\n" + marker)

dst.write_text(s)
print(f"  wrote {dst} ({len(s)} bytes, was {len(src.read_text())})")
PYEOF

echo
echo "=== sanity ==="
for f in prompts/api.prompt prompts/api_grounding.prompt prompts/api_depth.prompt; do
  printf "  %-32s %6s bytes  placeholder:%s\n" "$f" "$(wc -c < "$f")" \
    "$(grep -c INSERT_QUERY_HERE "$f")"
done
echo
echo "  anti-geometry line still present in:"
grep -l "left/right/up/down" prompts/*.prompt || echo "    (none)"
