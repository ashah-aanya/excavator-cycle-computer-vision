"""Tests for ``eval/clear_view.py``, the clear-view score experiment.

The score must fall when the picture around the arm's tip is blurred (dust), must not depend
on the video's pixel scale, and the AUC must read 1.0 for a perfect separator and 0.5 for none.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from excavator_cycles.kinematics import boom_base

EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))
spec = importlib.util.spec_from_file_location("clear_view", EVAL / "clear_view.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)

HEIGHT, WIDTH = 200, 320


def arm():
    core = np.zeros((HEIGHT, WIDTH), bool)
    core[110:150, 20:60] = True
    mask = core.copy()
    mask[118:130, 60:200] = True
    mask[104:138, 165:205] = True
    return mask, core


def textured(seed=0):
    """An image full of fine detail everywhere, so a blur visibly removes it."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def test_a_blurred_tip_scores_lower_than_a_sharp_one():
    mask, core = arm()
    pivot = boom_base(core)
    sharp = textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)  # a dust plume: detail is gone
    a = cv.clear_view(sharp, mask, core, pivot)
    b = cv.clear_view(blurred, mask, core, pivot)
    assert a is not None and b is not None
    assert a["sharp"] > 5 * b["sharp"]
    assert a["edges"] > 2 * b["edges"]


def test_the_score_does_not_depend_on_the_pixel_scale_of_the_video():
    mask, core = arm()
    pivot = boom_base(core)
    smooth = cv2.GaussianBlur(
        textured(1), (0, 0), 2
    )  # detail at a scale that survives resizing
    small = cv.clear_view(smooth, mask, core, pivot)
    factor = 2
    big_image = cv2.resize(smooth, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    big_mask = cv2.resize(
        mask.astype(np.uint8), None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    big_core = cv2.resize(
        core.astype(np.uint8), None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    big = cv.clear_view(big_image, big_mask, big_core, boom_base(big_core))
    assert abs(big["edges"] - small["edges"]) / small["edges"] < 0.35


def test_no_arm_means_no_score():
    empty = np.zeros((HEIGHT, WIDTH), bool)
    assert cv.clear_view(textured(), empty, empty, (10.0, 10.0)) is None


def test_auc_is_one_for_a_perfect_separator_half_for_none_and_none_without_both_groups():
    assert cv.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert cv.auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert cv.auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert cv.auc([], [0.3]) is None
    assert cv.auc([0.3], []) is None


def test_the_three_selection_rules_pick_what_they_say():
    frames = [
        dict(ext=10.0, sharp=1.0, edges=1.0, iou=0.1),
        dict(ext=50.0, sharp=2.0, edges=2.0, iou=0.2),  # the longest reach, a blurry view
        dict(
            ext=40.0, sharp=9.0, edges=9.0, iou=0.9
        ),  # a slightly shorter reach, a sharp view
        dict(ext=30.0, sharp=3.0, edges=3.0, iou=0.3),
        dict(ext=20.0, sharp=4.0, edges=4.0, iou=0.4),
        dict(ext=5.0, sharp=0.5, edges=0.5, iou=0.05),
    ]
    result = cv.selections(frames)
    assert result["S0 longest extension"] == 0.2
    assert result["S1 top-third extension, best sharp"] == 0.9
    assert result["S2 extension + sharp (rank sum)"] == 0.9
