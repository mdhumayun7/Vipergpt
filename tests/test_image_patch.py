"""Tests for ImagePatch geometry and free functions (pure python, no models)."""
import numpy as np

from vipergpt_repro.pipeline.image_patch import (
    ImagePatch,
    bool_to_yesno,
    coerce_to_numeric,
    distance,
)
from vipergpt_repro.pipeline.vision_bus import ModuleBus


def _img():
    return np.zeros((3, 100, 200), dtype="float32")


def test_full_patch_dimensions():
    p = ImagePatch(_img(), bus=ModuleBus(mock=True))
    assert p.left == 0 and p.lower == 0
    assert p.right == 200 and p.upper == 100
    assert p.horizontal_center == 100 and p.vertical_center == 50


def test_bool_to_yesno():
    assert bool_to_yesno(True) == "yes"
    assert bool_to_yesno(False) == "no"


def test_coerce_to_numeric():
    assert coerce_to_numeric("about 12 apples") == 12.0
    assert coerce_to_numeric("10-15") == 10.0


def test_distance_overlap_is_negative():
    bus = ModuleBus(mock=True)
    a = ImagePatch(_img(), 0, 0, 10, 10, bus=bus)
    b = ImagePatch(_img(), 0, 0, 10, 10, bus=bus)
    assert distance(a, b) < 0  # overlapping -> -IoU


def test_distance_disjoint_is_positive():
    bus = ModuleBus(mock=True)
    a = ImagePatch(_img(), 0, 0, 10, 10, bus=bus)
    b = ImagePatch(_img(), 20, 0, 30, 10, bus=bus)
    assert distance(a, b) == 10.0
