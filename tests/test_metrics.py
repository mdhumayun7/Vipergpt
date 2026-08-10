"""Unit tests for evaluators — checked against hand-computed values."""
from vipergpt_repro.eval import metrics


def test_iou_perfect_and_disjoint():
    assert metrics.iou_single([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert metrics.iou_single([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # two 10x10 boxes overlapping in a 10x5 region -> inter=50, union=150
    assert abs(metrics.iou_single([0, 0, 10, 10], [0, 5, 10, 15]) - (50 / 150)) < 1e-9


def test_refcoco_accuracy_thresholding():
    preds = [[0, 0, 10, 10], [0, 0, 10, 10]]
    gts = [[0, 0, 10, 10], [20, 20, 30, 30]]
    mean_iou, acc = metrics.refcoco_accuracy(preds, gts)
    assert abs(mean_iou - 0.5) < 1e-9
    assert acc == 0.5  # one IoU>0.7, one not


def test_vqa_normalize():
    assert metrics.vqa_normalize("The Red car.") == "red car"
    assert metrics.vqa_normalize("two") == "2"


def test_gqa_exact_match():
    assert metrics.gqa_accuracy(["a dog"], ["dog"]) == 1.0
    assert metrics.gqa_accuracy(["cat"], ["dog"]) == 0.0


def test_okvqa_soft_accuracy():
    # 3+ matching human answers -> score 1.0
    gts = [["dog"] * 3 + ["cat"] * 7]
    assert metrics.okvqa_accuracy(["dog"], gts) == 1.0
    # 1 match -> 1/3
    gts2 = [["dog"] + ["cat"] * 9]
    assert abs(metrics.okvqa_accuracy(["dog"], gts2) - 1 / 3) < 1e-9


def test_mc_accuracy():
    assert metrics.multiple_choice_accuracy([1, 2, 3], [1, 0, 3]) == 2 / 3
