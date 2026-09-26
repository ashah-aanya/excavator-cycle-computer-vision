"""Occupancy and the stable core -- the persistence trick the pipeline rests on.

These use synthetic masks -- a static body plus an arm that sweeps -- because
the property under test is "does persistence separate the two?", and only a
synthetic clip lets us state the right answer in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles import geometry
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


# -- otsu_threshold -------------------------------------------------------------
#
# These exist because `otsu_threshold` reached `main` with zero callers and zero
# tests while `architecture.html` claimed it was tested. `fsm.calibrate` then
# wired it in, so it went from dead to load-bearing without ever gaining one.


def test_otsu_lands_between_the_two_populations_however_far_apart_they_are():
    """The defect: a wide empty gap makes every split in it score identically.

    Otsu's objective is `w_low * w_high * (mean_low - mean_high)**2`. Once the
    split is anywhere inside an empty gap, all three terms stop changing, so the
    whole gap ties -- and `argmax` breaks a tie at the LOWEST index. The
    threshold therefore snaps to the bottom of the gap, hugging the lower
    population, and it gets worse the cleaner the separation. On this clip the
    gap is 1-2 bins wide so the error is invisible; on a video where the dig
    depth and carry height are cleanly separated it is most of the gap.
    """
    rng = np.random.default_rng(0)
    for separation in (4.0, 20.0, 100.0):
        values = np.concatenate(
            [rng.normal(0.0, 1.0, 20_000), rng.normal(separation, 1.0, 20_000)]
        )
        threshold = geometry.otsu_threshold(values)
        # Generous: anywhere in the middle half of the gap. The point is that it
        # tracks the separation instead of sticking near the lower hump.
        assert separation * 0.25 < threshold < separation * 0.75, (
            f"separation {separation}: threshold {threshold:.3f} is not between the humps"
        )


def test_otsu_splits_two_populations_of_very_different_size():
    """A tie also spans the gap when the groups are lopsided, and `w_low *
    w_high` is then strongly asymmetric -- so this is where a midpoint rule
    could plausibly go wrong even though the equal-size case passes."""
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0.0, 1.0, 38_000), rng.normal(30.0, 1.0, 2_000)])
    threshold = geometry.otsu_threshold(values)
    assert 7.5 < threshold < 22.5, threshold


def test_otsu_on_a_constant_signal_returns_the_constant():
    """Degenerate, but it must not crash or return a NaN: there is no split."""
    assert geometry.otsu_threshold(np.full(500, 0.4)) == pytest.approx(0.4)
