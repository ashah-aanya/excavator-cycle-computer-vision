"""The cabin reference box.

These use synthetic masks -- a static body plus an arm that sweeps -- because
the property under test is "does persistence separate the two?", and only a
synthetic clip lets us state the right answer in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.cabin import (
    cabin_box,
    occupancy,
    rolling_boxes,
    stability,
    stable_core,
)

BODY = (slice(60, 90), slice(20, 50))  # rows 60-89, cols 20-49: never moves


def _clip(n: int = 40) -> list[np.ndarray]:
    """A static body, plus a one-pixel-wide arm that sweeps a wide arc.

    The arm visits each column exactly once, so every arm pixel has occupancy
    1/n while every body pixel has occupancy 1.0. That gap is what the cabin
    box is supposed to exploit.
    """
    masks = []
    for index in range(n):
        frame = np.zeros((100, 100), dtype=bool)
        frame[BODY] = True
        frame[20:60, 50 + index] = True  # the arm, at a different column each frame
        masks.append(frame)
    return masks


def test_occupancy_is_a_fraction_of_frames():
    occ = occupancy(_clip(n=40))
    assert occ[70, 30] == pytest.approx(1.0)  # body: present every frame
    assert occ[30, 50] == pytest.approx(1 / 40)  # arm: present once
    assert occ[5, 5] == pytest.approx(0.0)  # sky: never


def test_occupancy_rejects_an_empty_clip():
    with pytest.raises(ValueError, match="no masks"):
        occupancy([])


def test_stable_core_keeps_the_body_and_drops_the_sweeping_arm():
    core, threshold = stable_core(_clip(), quantile=0.90)
    assert core[BODY].all(), "every body pixel should survive"
    assert not core[30, 50], "a pixel the arm visited once should not"
    # The cutoff is derived from this clip, not hardcoded; it must land above
    # the arm's occupancy and at or below the body's.
    assert 1 / 40 < threshold <= 1.0


def test_cabin_box_encloses_the_body_only():
    fitted = cabin_box(_clip(), quantile=0.90)
    assert fitted.box == (20.0, 60.0, 49.0, 89.0)  # cols 20-49, rows 60-89
    assert fitted.centroid == pytest.approx((34.5, 74.5))
    assert fitted.core_pixels == 30 * 30
    assert fitted.components == 1, "a clean body should be one blob"


def test_right_and_bottom_are_exposed_because_cues_should_use_them():
    fitted = cabin_box(_clip(), quantile=0.90)
    assert fitted.right == fitted.box[2]
    assert fitted.bottom == fitted.box[3]


def test_a_split_core_is_reported_not_silently_cleaned():
    """Two disconnected persistent blobs must surface, not be filtered away."""
    masks = []
    for _ in range(10):
        frame = np.zeros((100, 100), dtype=bool)
        frame[BODY] = True
        frame[10:20, 80:90] = True  # a second, disconnected persistent region
        masks.append(frame)
    assert cabin_box(masks, quantile=0.5).components == 2


def test_stability_reports_both_sensitivities():
    report = stability(_clip(), quantile=0.90, window_seconds=1.0)
    assert set(report["quantile_edge_range"]) == {"left", "top", "right", "bottom"}
    assert set(report["rolling_edge_std"]) == {"left", "top", "right", "bottom"}
    assert report["rolling_count"] > 0


def test_rolling_window_length_is_seconds_not_frames():
    """The same clip at two sample rates must use the same amount of time.

    This is the frame-rate independence rule: a hidden video at a different
    rate must get the same physical window, not the same number of samples.
    """
    masks = _clip(n=40)
    slow = stability(
        masks, 0.90, times_seconds=[i * 0.1 for i in range(40)], window_seconds=1.0
    )
    fast = stability(
        masks, 0.90, times_seconds=[i * 0.2 for i in range(40)], window_seconds=1.0
    )
    assert slow["rolling_window_samples"] == 10  # 1.0s / 0.1s
    assert fast["rolling_window_samples"] == 5  # 1.0s / 0.2s


def test_rolling_boxes_are_centred_and_none_at_the_edges():
    boxes = rolling_boxes(_clip(n=20), quantile=0.90, window=5)
    assert len(boxes) == 20
    assert boxes[0] is None and boxes[-1] is None, "window runs off the ends"
    assert boxes[10] is not None
