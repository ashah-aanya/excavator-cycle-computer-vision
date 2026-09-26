"""Occupancy and the stable core -- the persistence trick the pipeline rests on.

These use synthetic masks -- a static body plus an arm that sweeps -- because
the property under test is "does persistence separate the two?", and only a
synthetic clip lets us state the right answer in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.geometry import occupancy, stable_core

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
