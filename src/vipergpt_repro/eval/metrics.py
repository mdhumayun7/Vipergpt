"""Evaluation metrics, ported to match the official viper evaluators.

- IoU / [email protected] for RefCOCO (boxes in bottom-left (left,lower,right,upper)).
- VQA-style answer post-processing + exact match for GQA.
- VQA soft accuracy (10-annotator) for OK-VQA.
- Multiple-choice accuracy for NExT-QA.
"""
from __future__ import annotations

import re

# ---- VQA answer normalization (from the official VQA eval, used by GQA/OK-VQA) ----
_CONTRACTIONS = {"aint": "ain't", "dont": "don't", "cant": "can't", "wont": "won't"}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (re.search(_COMMA_STRIP, text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _PERIOD_STRIP.sub("", out)
    return out


def _process_digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = _MANUAL_MAP.get(w, w)
        if w not in _ARTICLES:
            words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_normalize(answer: str) -> str:
    a = (answer or "").replace("\n", " ").replace("\t", " ").strip().lower()
    a = _process_punctuation(a)
    a = _process_digit_article(a)
    return a.strip()


# ---------------------------------------------------------------- RefCOCO IoU
def _to_box(p):
    """Coerce prediction to [left, lower, right, upper]."""
    if p is None:
        return None
    if hasattr(p, "left"):
        return [p.left, p.lower, p.right, p.upper]
    if isinstance(p, (list, tuple)):
        return list(p)
    return None


def iou_single(pred_box, gt_box) -> float:
    if pred_box is None:
        return 0.0
    ax1, ay1, ax2, ay2 = pred_box
    bx1, by1, bx2, by2 = gt_box
    inter = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(min(ay2, by2) - max(ay1, by1), 0)
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def refcoco_accuracy(predictions, ground_truths, acc_thresh: float = 0.7):
    """Returns (mean_iou, [email protected]) — matches official refcoco.py."""
    assert len(predictions) == len(ground_truths)
    total_iou, hits, n = 0.0, 0, 0
    for p, g in zip(predictions, ground_truths):
        box = _to_box(p)
        i = iou_single(box, g) if box is not None else 0.0
        total_iou += i
        hits += int(i > acc_thresh)
        n += 1
    n = max(n, 1)
    return total_iou / n, hits / n


# ---------------------------------------------------------------- GQA exact match
def gqa_accuracy(predictions, ground_truths) -> float:
    if not predictions:
        return 0.0
    return sum(vqa_normalize(p) == vqa_normalize(g) for p, g in zip(predictions, ground_truths)) / len(predictions)


# ---------------------------------------------------------------- OK-VQA soft acc
def okvqa_accuracy(predictions, gt_answer_lists) -> float:
    """gt_answer_lists: list of lists of the 10 human answers per question."""
    if not predictions:
        return 0.0
    total = 0.0
    for p, gts in zip(predictions, gt_answer_lists):
        pn = vqa_normalize(p)
        matches = sum(pn == vqa_normalize(g) for g in gts)
        total += min(1.0, matches / 3.0)
    return total / len(predictions)


# ---------------------------------------------------------------- NExT-QA MC
def multiple_choice_accuracy(pred_indices, gt_indices) -> float:
    if not pred_indices:
        return 0.0
    return sum(int(p) == int(g) for p, g in zip(pred_indices, gt_indices)) / len(pred_indices)
