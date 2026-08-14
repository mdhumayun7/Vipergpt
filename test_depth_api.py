"""Unit tests for the depth primitives, using stub patches with KNOWN depths.

No model, no GPU, no images. The point is to pin the SIGN and the ordering semantics
in a test that fails loudly if either flips again — three sign errors in two days all
came from ordering code, and none of them were covered by a test.

    python -B test_depth_api.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from vipergpt_repro.pipeline.image_patch import (  # noqa: E402
    depth_order, distance_3d, is_behind, is_between_3d, is_in_front_of,
)


class StubPatch:
    """Minimal stand-in: fixed depth and centre, no model behind it."""

    def __init__(self, name, depth, hc=0.0, vc=0.0):
        self.name = name
        self._depth = depth
        self.horizontal_center = hc
        self.vertical_center = vc

    def compute_depth(self):
        if self._depth is None:
            raise ValueError("undefined depth")
        return self._depth

    def __repr__(self):
        return self.name


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"         got  {got}")
        print(f"         want {want}")
    return ok


def main():
    print("=" * 62)
    print("  DEPTH PRIMITIVES — UNIT TESTS")
    print("=" * 62)
    print("  Convention under test: compute_depth() LARGER = FURTHER.\n")

    near = StubPatch("near", 1.0)
    mid = StubPatch("mid", 5.0)
    far = StubPatch("far", 9.0)
    results = []

    print("depth_order")
    results.append(check("sorts near -> far",
                         depth_order([far, near, mid]), [near, mid, far]))
    results.append(check("reverse=True gives far -> near",
                         depth_order([near, far, mid], reverse=True), [far, mid, near]))
    results.append(check("[0] is the CLOSEST (backs 'closest'/'nearest')",
                         depth_order([far, mid, near])[0], near))
    results.append(check("[-1] is the FURTHEST (backs 'farthest')",
                         depth_order([near, mid, far])[-1], far))
    results.append(check("empty input -> empty output", depth_order([]), []))
    results.append(check("single patch passes through", depth_order([mid]), [mid]))

    bad = StubPatch("bad", None)
    results.append(check("undefined depth is dropped, not placed at an end",
                         depth_order([far, bad, near]), [near, far]))

    print("\nis_behind / is_in_front_of")
    results.append(check("far is behind near", is_behind(far, near), True))
    results.append(check("near is not behind far", is_behind(near, far), False))
    results.append(check("near is in front of far", is_in_front_of(near, far), True))
    results.append(check("margin suppresses a small difference",
                         is_behind(StubPatch("a", 5.2), mid, margin=1.0), False))
    results.append(check("margin allows a large difference",
                         is_behind(far, mid, margin=1.0), True))

    print("\nis_between_3d")
    results.append(check("mid is between near and far",
                         is_between_3d(mid, near, far), True))
    results.append(check("argument order does not matter",
                         is_between_3d(mid, far, near), True))
    results.append(check("near is not between mid and far",
                         is_between_3d(near, mid, far), False))

    print("\ndistance_3d")
    a = StubPatch("a", 0.0, hc=0.0, vc=0.0)
    b = StubPatch("b", 3.0, hc=4.0, vc=0.0)
    results.append(check("3-4-5 triangle in (x, depth)", round(distance_3d(a, b), 6), 5.0))
    results.append(check("symmetric",
                         round(distance_3d(a, b), 6), round(distance_3d(b, a), 6)))
    results.append(check("zero to itself", distance_3d(a, a), 0.0))

    print("\n" + "=" * 62)
    n_ok = sum(results)
    print(f"  {n_ok}/{len(results)} passed")
    if n_ok != len(results):
        print("\n  A FAILURE HERE MEANS THE DEPTH SIGN OR ORDERING HAS FLIPPED.")
        print("  Fix before running anything that consumes these primitives.")
    print("=" * 62)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
