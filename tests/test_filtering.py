"""Tests for the temporal filter (design doc §4.3).

The behaviour under test is a judgement, not just arithmetic: which measurements
to believe. So the cases are the ones that distinguish a useful filter from a
harmful one --

* a single-frame jump to the counterweight must be rejected (what §4.3 asks for)
* a genuine fast swing must NOT be rejected, or the filter would erase the
  motion that the swing boundary is detected from
* a gap with no measurement must coast rather than invent a position
"""

from __future__ import annotations

import numpy as np

from excavator_cycles.filtering import constant_velocity_filter, filter_track

DT = 0.1
NOISE = 0.01
GATE = 3.0


def smooth_track(n=60, speed=0.02):
    """A bucket moving steadily, as it would during a swing."""
    return np.stack([np.arange(n) * speed, np.full(n, 0.5)], axis=1)


# --- the case §4.3 names ----------------------------------------------------


def test_single_frame_jump_is_rejected():
    """One frame where the fit lands on the counterweight instead of the bucket."""
    track = smooth_track()
    track[30] = [5.0, 5.0]  # far away, for one frame only

    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)

    assert not result.accepted[30], "the jump should not be believed"
    assert result.accepted[29] and result.accepted[31], "its neighbours should be"
    # And the filtered position must stay on the real trajectory.
    assert np.hypot(*(result.positions[30] - [30 * 0.02, 0.5])) < 0.1


def test_the_jump_does_not_contaminate_later_samples():
    """Rejecting is only useful if the outlier leaves no trace."""
    track = smooth_track()
    track[30] = [5.0, 5.0]

    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    for index in (35, 40, 50):
        expected = np.array([index * 0.02, 0.5])
        assert np.hypot(*(result.positions[index] - expected)) < 0.05


def test_genuine_fast_motion_is_kept():
    """A swing is fast. A filter that rejects it would erase the T4 boundary."""
    n = 60
    track = np.stack([np.linspace(0, 3.0, n), np.full(n, 0.5)], axis=1)  # 0.5 units/s

    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    assert result.accepted[5:].mean() > 0.95, "steady fast motion must be believed"


def test_acceleration_is_tolerated():
    """The bucket starts and stops constantly; that is not an outlier."""
    n = 80
    t = np.arange(n) * DT
    track = np.stack([0.5 * t**2 * 0.1, np.full(n, 0.5)], axis=1)

    result = constant_velocity_filter(track, DT, 1.0, NOISE, GATE)
    assert result.accepted[5:].mean() > 0.9


# --- gaps -------------------------------------------------------------------


def test_missing_measurements_are_coasted():
    """No measurement is not the same as a measurement of nothing."""
    track = smooth_track()
    track[20:25] = np.nan

    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    assert np.all(np.isfinite(result.positions[20:25])), "the gap must still be filled"
    assert not result.accepted[20:25].any(), "but not marked as measured"
    # Coasting on constant velocity should stay close through a short gap.
    assert np.hypot(*(result.positions[24] - [24 * 0.02, 0.5])) < 0.05


def test_track_starting_with_gaps():
    track = smooth_track()
    track[:5] = np.nan
    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    assert not np.isfinite(result.positions[0]).any(), "nothing to say before the first fix"
    assert np.isfinite(result.positions[10]).all()


def test_all_missing():
    track = np.full((20, 2), np.nan)
    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    assert not result.accepted.any()
    assert result.rejection_rate == 0.0, "nothing measured means nothing rejected"


# --- scale independence -----------------------------------------------------


def test_filtering_is_scale_invariant():
    """The same motion at two camera distances must be judged identically.

    This is why the track is normalised by the machine's reach before filtering:
    one set of noise parameters then serves any video.
    """
    points = [(x, 100.0) for x in np.arange(0, 60, 1.0)]
    points[30] = (400.0, 400.0)  # the jump

    near = filter_track(
        points, DT, scale=100.0, process_noise=0.5, measurement_noise=NOISE, gate_sigma=GATE
    )
    far = filter_track(
        [(x * 3, y * 3) for x, y in points],
        DT,
        scale=300.0,
        process_noise=0.5,
        measurement_noise=NOISE,
        gate_sigma=GATE,
    )
    assert np.array_equal(near.accepted, far.accepted)


def test_none_entries_are_treated_as_gaps():
    points: list[tuple[float, float] | None] = [(x, 50.0) for x in np.arange(0, 30, 1.0)]
    points[10] = None
    result = filter_track(points, DT, 100.0, 0.5, NOISE, GATE)
    assert not result.accepted[10]
    assert np.isfinite(result.positions[10]).all()


def test_rejection_rate_counts_only_measured_samples():
    track = smooth_track(40)
    track[10] = [9.0, 9.0]
    track[20:25] = np.nan
    result = constant_velocity_filter(track, DT, 0.5, NOISE, GATE)
    assert 0 < result.rejection_rate < 0.1
