"""Tests for fitting the kinematic chain to a silhouette.

The synthetic machine here is a body with a two-segment arm whose pose is known,
so the tests can assert that the fit recovers the arm's *end* rather than
whatever pixel happens to be farthest away.

The central case is ``test_finds_the_tip_when_the_arm_folds_back``: a raised arm
whose elbow is farther from the base than its tip. Straight-line distance picks
the elbow; geodesic distance picks the tip. That is the exact failure this
module exists to fix, and it was visible on the real video as a marker sitting
on top of the boom throughout hauling.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from excavator_cycles.kinematics import (
    body_core,
    boom_base,
    fit_chain,
    fit_frame,
    geodesic_distance,
    trace_path,
)


def draw_arm(shape=(200, 300), base=(60, 120), joints=((120, 60), (200, 100)), width=7):
    """A body blob plus a two-segment arm, as a binary mask."""
    import cv2

    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, base, 22, 1, -1)  # body
    points = [base, *joints]
    for start, end in pairwise(points):
        cv2.line(mask, start, end, 1, width)
    return mask.astype(bool)


# --- geodesic distance ------------------------------------------------------


def test_geodesic_distance_follows_the_mask():
    """Distance must travel along the shape, not across the gap."""
    mask = np.zeros((50, 50), bool)
    mask[10, 5:45] = True  # along the top
    mask[10:40, 44] = True  # then down the right side

    distance = geodesic_distance(mask, (5, 10))
    corner = distance[10, 44]
    bottom = distance[39, 44]
    assert corner > 35, "must traverse the horizontal run"
    assert bottom > corner, "and then continue down"
    # Straight-line distance from the seed to the bottom-right is ~48; the path
    # is ~68. The difference is the point.
    assert bottom > np.hypot(44 - 5, 39 - 10)


def test_unreached_pixels_are_marked():
    mask = np.zeros((20, 20), bool)
    mask[2:6, 2:6] = True
    mask[15:18, 15:18] = True  # a separate island
    distance = geodesic_distance(mask, (3, 3))
    assert distance[16, 16] == -1
    assert distance[4, 4] >= 0


def test_seed_outside_the_mask_still_works():
    """A base estimated slightly off the silhouette must not fail the frame."""
    mask = np.zeros((40, 40), bool)
    mask[20:24, 10:30] = True
    distance = geodesic_distance(mask, (5, 5))
    assert (distance >= 0).sum() > 0


# --- the failure this module fixes -----------------------------------------


def test_finds_the_tip_when_the_arm_folds_back():
    """The case that broke the old approach.

    The arm rises steeply then comes back down, so the ELBOW is farther from the
    base in a straight line than the TIP is. Geodesic distance still puts the tip
    at the end, because reaching it means walking the whole arm.
    """
    base = (60, 150)
    elbow = (150, 30)  # high and far
    tip = (170, 110)  # beyond the elbow along the arm, but nearer the base
    mask = draw_arm(shape=(200, 300), base=base, joints=(elbow, tip))

    straight_to_elbow = np.hypot(elbow[0] - base[0], elbow[1] - base[1])
    straight_to_tip = np.hypot(tip[0] - base[0], tip[1] - base[1])
    assert straight_to_elbow > straight_to_tip, "fixture must reproduce the trap"

    distance = geodesic_distance(mask, base)
    end = np.unravel_index(int(np.argmax(distance)), distance.shape)
    found = (end[1], end[0])

    assert np.hypot(found[0] - tip[0], found[1] - tip[1]) < 20, (
        f"geodesic-farthest point {found} should be the tip {tip}, not the elbow {elbow}"
    )


def test_straight_line_would_have_picked_the_elbow():
    """Documents the old behaviour, so the regression is explicit."""
    base = (60, 150)
    elbow, tip = (150, 30), (170, 110)
    mask = draw_arm(shape=(200, 300), base=base, joints=(elbow, tip))

    ys, xs = np.nonzero(mask)
    farthest = int(np.argmax((xs - base[0]) ** 2 + (ys - base[1]) ** 2))
    picked = (xs[farthest], ys[farthest])
    assert np.hypot(picked[0] - elbow[0], picked[1] - elbow[1]) < 25, (
        "straight-line distance picks the elbow -- which is why it was wrong"
    )


# --- chain fitting ----------------------------------------------------------


def test_fit_recovers_four_points_in_order():
    mask = draw_arm(base=(60, 120), joints=((140, 60), (210, 110)))
    chain = fit_frame(mask, (60, 120))
    assert chain is not None
    assert len(chain.points) == 4
    # The path runs base -> tip, so successive points get farther along the arm.
    assert chain.path_length > 0
    assert chain.residual < 6, "a three-link model should describe a two-segment arm"


def test_fit_locates_the_bend():
    """The middle points should land near the real joint, not anywhere."""
    elbow = (140, 60)
    mask = draw_arm(base=(60, 120), joints=(elbow, (210, 110)))
    chain = fit_frame(mask, (60, 120))
    assert chain is not None
    nearest = min(
        np.hypot(p[0] - elbow[0], p[1] - elbow[1])
        for p in (chain.boom_stick, chain.stick_bucket)
    )
    assert nearest < 25, f"no fitted joint near the real bend at {elbow}"


def test_link_lengths_are_reported():
    mask = draw_arm()
    chain = fit_frame(mask, (60, 120))
    assert chain is not None
    lengths = chain.link_lengths()
    assert len(lengths) == 3
    assert all(length > 0 for length in lengths)


def test_fit_rejects_a_tiny_path():
    assert fit_chain([(0, 0), (1, 1)]) is None


def test_fit_frame_on_an_empty_mask():
    assert fit_frame(np.zeros((20, 20), bool), (5, 5)) is None


# --- body and base ----------------------------------------------------------


def test_body_core_is_what_persists():
    """The body is in every frame; the arm sweeps and is not."""
    frames = []
    for offset in range(0, 60, 6):
        mask = np.zeros((120, 200), bool)
        mask[70:100, 40:80] = True  # body, always here
        mask[30 + offset : 40 + offset, 80:160] = True  # arm, moving
        frames.append(mask)

    core = body_core(frames, quantile=0.9)
    assert core[80, 60], "the body must survive"
    assert not core[35, 120], "a swept-through position must not"


def test_boom_base_sits_at_the_top_of_the_body():
    """Anchored at the cab, not down among the tracks."""
    core = np.zeros((120, 200), bool)
    core[60:110, 40:90] = True
    x, y = boom_base(core)
    assert 40 <= x <= 90
    assert y < 75, f"base at y={y} should be in the upper part of the body (60-110)"


def test_boom_base_needs_a_body():
    with pytest.raises(ValueError, match="body core is empty"):
        boom_base(np.zeros((10, 10), bool))


def test_trace_path_runs_base_to_tip():
    mask = np.zeros((40, 60), bool)
    mask[20, 5:50] = True
    distance = geodesic_distance(mask, (5, 20))
    path = trace_path(distance, (20, 49))
    assert path[0][0] < path[-1][0], "path must be ordered from the seed outward"
    assert len(path) > 40
