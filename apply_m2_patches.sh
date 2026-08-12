#!/usr/bin/env bash
# Two edits needed before the executor can produce correct numbers.
#   1. find() drops the crop offset — boxes from a nested patch land in the wrong
#      frame. Invisible on root-level programs, wrong on any program that crops.
#   2. Wire the CLIP backend into ModuleBus in place of the X-VLM stub.
# Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path

# ---------------------------------------------------------------- 1. find() offset
p = Path("src/vipergpt_repro/pipeline/image_patch.py")
s = p.read_text()

old = '''    def find(self, object_name: str) -> list[ImagePatch]:
        boxes = self._bus.call("find", self.cropped_image, object_name)
        patches = []
        for (left, lower, right, upper) in boxes:
            # boxes are relative to this crop; offset back into this patch's frame
            patches.append(
                ImagePatch(
                    self.cropped_image,
                    int(left),
                    int(lower),
                    int(right),
                    int(upper),
                    bus=self._bus,
                )
            )
        return patches'''

new = '''    def find(self, object_name: str) -> list[ImagePatch]:
        boxes = self._bus.call("find", self.cropped_image, object_name)
        patches = []
        for (left, lower, right, upper) in boxes:
            # The detector sees only `cropped_image`, so its boxes are in CROP
            # coordinates. Child patches are constructed from `cropped_image` with
            # those crop-relative bounds (so the pixel slice is right), then their
            # reported coordinates are shifted into THIS patch's frame. Without the
            # shift, any program that crops before finding reports boxes in the
            # wrong coordinate space — silently, and only for nested calls.
            child = ImagePatch(
                self.cropped_image,
                int(left),
                int(lower),
                int(right),
                int(upper),
                bus=self._bus,
            )
            child.left += self.left
            child.right += self.left
            child.lower += self.lower
            child.upper += self.lower
            child.horizontal_center = (child.left + child.right) / 2
            child.vertical_center = (child.lower + child.upper) / 2
            patches.append(child)
        return patches'''

assert old in s, "find() anchor not found — file already modified?"
p.write_text(s.replace(old, new))
print("[1/2] patched find() to offset child coordinates")

# ---------------------------------------------------------------- 2. CLIP in the bus
p = Path("src/vipergpt_repro/pipeline/vision_bus.py")
s = p.read_text()

old_imp = "from vipergpt_repro.models import blip2, codegen, depth, glip, text_llm, xvlm  # noqa: F401"
new_imp = ("from vipergpt_repro.models import (  # noqa: F401\n"
           "            blip2, clip_vlm, codegen, depth, glip, text_llm, xvlm,\n"
           "        )")
assert old_imp in s, "import anchor not found"
s = s.replace(old_imp, new_imp)

old_x = '''        if lm.get("xvlm", False):
            x = xvlm.XVLMModel(self.cfg)
            self.register("verify_property", x.verify_property)
            self.register("best_text_match", x.best_text_match)
            self.register("best_image_match", x.best_image_match)
            self._loaded["xvlm"] = x'''
new_x = '''        if lm.get("xvlm", False):
            x = xvlm.XVLMModel(self.cfg)
            self.register("verify_property", x.verify_property)
            self.register("best_text_match", x.best_text_match)
            self.register("best_image_match", x.best_image_match)
            self._loaded["xvlm"] = x
        elif lm.get("clip", False):
            # DEVIATION D9: CLIP substitutes for X-VLM on verify_property /
            # best_text_match / best_image_match. See models/clip_vlm.py.
            c = clip_vlm.CLIPModel_(self.cfg)
            self.register("verify_property", c.verify_property)
            self.register("best_text_match", c.best_text_match)
            self.register("best_image_match", c.best_image_match)
            self._loaded["clip"] = c
            logger.warning("Using CLIP for verify_property/best_match (D9): X-VLM not built.")'''
assert old_x in s, "xvlm block anchor not found"
s = s.replace(old_x, new_x)
p.write_text(s)
print("[2/2] wired CLIP into ModuleBus as the X-VLM substitute")
PYEOF

echo
echo "verifying syntax..."
python -c "import ast; [ast.parse(open(f).read()) for f in [
  'src/vipergpt_repro/pipeline/image_patch.py',
  'src/vipergpt_repro/pipeline/vision_bus.py',
  'src/vipergpt_repro/models/clip_vlm.py',
  'src/vipergpt_repro/eval/execute_refcoco.py']]" && echo "SYNTAX OK"

echo
echo "Next: download CLIP on the LOGIN node (compute nodes are offline):"
echo "  hf download openai/clip-vit-large-patch14"
