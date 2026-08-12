"""ImagePatch and free functions — the API surface generated programs call.

Ported from the official viper `image_patch.py` semantics (see prompts/api.prompt),
but the actual model calls are dispatched through a `ModuleBus` so we can run with
either real models or CPU mocks. Programs never see the bus directly; they use the
same method names documented in the API prompt.

Coordinate convention (MUST match the RefCOCO evaluator): boxes are
(left, lower, right, upper) with a BOTTOM-LEFT origin. `cropped_image` is a
channel-first float tensor [3, H, W]. `lower` < `upper` in this convention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # avoid importing torch at module import for CPU-only smoke
    pass


class ImagePatch:
    def __init__(
        self,
        image,
        left: int | None = None,
        lower: int | None = None,
        right: int | None = None,
        upper: int | None = None,
        bus=None,
    ):
        # `bus` carries the module dispatcher; propagated to child patches.
        self._bus = bus

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]
        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    # ---- perception API (dispatched through the bus) ----

    def find(self, object_name: str) -> list[ImagePatch]:
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
        return patches

    def exists(self, object_name: str) -> bool:
        return len(self.find(object_name)) > 0

    def verify_property(self, object_name: str, visual_property: str) -> bool:
        return bool(self._bus.call("verify_property", self.cropped_image, object_name, visual_property))

    def best_text_match(self, option_list: list[str], prefix: str | None = None) -> str:
        return self._bus.call("best_text_match", self.cropped_image, option_list, prefix)

    def simple_query(self, question: str | None = None) -> str:
        return self._bus.call("simple_query", self.cropped_image, question)

    def compute_depth(self) -> float:
        return float(self._bus.call("compute_depth", self.cropped_image))

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return self._bus.call("llm_query", question, long_answer)

    # ---- geometry helpers (pure python) ----

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        return ImagePatch(self.cropped_image, left, lower, right, upper, bus=self._bus)

    def overlaps_with(self, left, lower, right, upper) -> bool:
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def __repr__(self) -> str:
        return f"ImagePatch(left={self.left}, lower={self.lower}, right={self.right}, upper={self.upper})"


def best_image_match(list_patches: list[ImagePatch], content: list[str], return_index: bool = False):
    if not list_patches:
        return None
    bus = list_patches[0]._bus
    idx = bus.call("best_image_match", [p.cropped_image for p in list_patches], content)
    return idx if return_index else list_patches[idx]


def distance(patch_a: ImagePatch, patch_b: ImagePatch) -> float:
    """Edge distance between two patches; negative (=-IoU) if they overlap."""
    a_l, a_lo, a_r, a_u = patch_a.left, patch_a.lower, patch_a.right, patch_a.upper
    b_l, b_lo, b_r, b_u = patch_b.left, patch_b.lower, patch_b.right, patch_b.upper

    # horizontal / vertical gaps (0 if overlapping on that axis)
    dx = max(a_l - b_r, b_l - a_r, 0)
    dy = max(a_lo - b_u, b_lo - a_u, 0)

    if dx == 0 and dy == 0:
        # overlapping -> negative IoU
        inter_l, inter_r = max(a_l, b_l), min(a_r, b_r)
        inter_lo, inter_u = max(a_lo, b_lo), min(a_u, b_u)
        inter = max(inter_r - inter_l, 0) * max(inter_u - inter_lo, 0)
        area_a = (a_r - a_l) * (a_u - a_lo)
        area_b = (b_r - b_l) * (b_u - b_lo)
        union = area_a + area_b - inter
        return -inter / union if union > 0 else 0.0
    return (dx**2 + dy**2) ** 0.5


def bool_to_yesno(bool_answer: bool) -> str:
    return "yes" if bool_answer else "no"


def coerce_to_numeric(string: str) -> float:
    """Strip non-numeric chars; for a range like '10-15' return the first value."""
    import re

    string = string.replace(",", "")
    m = re.search(r"-?\d+\.?\d*", string)
    return float(m.group()) if m else float("nan")


Number = Union[int, float]
