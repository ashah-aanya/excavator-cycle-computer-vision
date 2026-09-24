"""Tests for onset detection -- finding the moment a signal leaves rest.

The distinction these tests exist to protect
--------------------------------------------
A **level** (a height, a bearing) has a value even when nothing is moving, and
that value means nothing on another video -- so its rest must be estimated.

A **rate** does not. Not moving is ``0``: absolute, identical on every video
ever shot. Estimating it can only introduce error, and on this project it did
-- the estimator settled on the hauling rate instead of the dig plateau and
every transition measured from it was meaningless.

So ``motion_boundary`` fixes the level at zero and measures only the noise
width, while ``rest_boundary`` estimates both. Each test below pins one of the
specific failures that taught us the difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.onsets import (
    motion_boundary,
    noise_scale,
)

# --- rest is zero, for a rate --------------------------------------------


def test_a_rate_needs_no_estimated_rest_level():
    """The point of `motion_boundary`: not moving is 0, on every video.

    A rate that sits at zero, moves, and stops. The onset must be found
    WITHOUT the function ever inferring what "rest" looks like -- so this
    fixture deliberately spends most of its span moving, which is exactly the
    case where estimating a level picks the moving value and fails.
    """
    times = np.arange(0, 20, 0.1)
    rate = np.zeros_like(times)
    rate[times >= 5.0] = 1.0  # moves from 5 s onward: 75% of the clip
    rate += np.random.default_rng(0).normal(0, 0.01, len(times))

    found = motion_boundary(rate, times, (0.0, 20.0), mode="departs")
    assert found is not None
    assert abs(found - 5.0) <= 0.3, f"onset found at {found:.2f}, expected 5.0"


def test_the_noise_scale_is_the_noise_not_the_signal_magnitude():
    """A rate's own MAD measures its typical MOTION, not its noise.

    Measured on the development video: MAD over the whole of `dh/dt` was
    0.0420 L/s while the real noise floor was ~0.004 -- wide enough to swallow
    the entire hauling lift and report no event at all.
    """
    times = np.arange(0, 20, 0.1)
    rate = np.zeros_like(times)
    rate[times >= 4.0] = 0.5
    rate[times >= 12.0] = -0.5
    rate += np.random.default_rng(1).normal(0, 0.005, len(times))

    assert noise_scale(rate) < 0.05, "the estimator is reading the motion"
    assert motion_boundary(rate, times, (0.0, 11.0), mode="departs") is not None


def test_noise_scale_recovers_a_known_sigma_whatever_the_signal_does():
    """The property the quiet-window estimator lacked: a STABLE bias.

    That one returned 0.46x to 0.80x of the true sigma depending on the window
    and on how much of the clip was moving. This one holds ~0.85x across a
    step, a ramp, and a signal moving for 90% of its length -- so one `sigma`
    means the same thing on a digging plateau and on a full-speed slew.
    """
    times = np.arange(0, 20, 0.1)
    rng = np.random.default_rng(0)
    sigma = 0.01
    shapes = {
        "step": np.where(times >= 5.0, 1.0, 0.0),
        "ramp": 0.2 * times,
        "moving 90% of the clip": np.where(times >= 2.0, 1.0, 0.0),
    }
    for name, base in shapes.items():
        estimated = noise_scale(base + rng.normal(0, sigma, len(times)))
        assert 0.7 * sigma <= estimated <= 1.2 * sigma, (
            f"on a {name} the estimate was {estimated:.5f}, not near {sigma}"
        )


def test_arrives_finds_where_motion_stopped():
    times = np.arange(0, 20, 0.1)
    rate = np.zeros_like(times)
    rate[(times >= 3.0) & (times < 9.0)] = -0.8
    rate += np.random.default_rng(2).normal(0, 0.01, len(times))

    found = motion_boundary(rate, times, (0.0, 20.0), mode="arrives")
    assert found is not None
    assert abs(found - 9.0) <= 0.3, f"settled at {found:.2f}, expected 9.0"


def test_a_rate_that_never_rests_returns_none_not_the_bracket_edge():
    """The failure that masqueraded as this project's first passing transition.

    When the walk reaches the bracket edge without finding rest, the old code
    returned `times[limit]` -- the bracket's own opening time, reported as a
    measured onset. It scored well only because the bracket opened near the
    truth.
    """
    times = np.arange(0, 20, 0.1)
    rate = np.full_like(times, 2.0)  # moving for the entire bracket
    rate += np.random.default_rng(3).normal(0, 0.01, len(times))

    found = motion_boundary(rate, times, (5.0, 15.0), mode="departs")
    assert found is None or found > 5.0 + 1e-9, (
        f"returned {found}, which is the bracket edge dressed as a measurement"
    )


def test_motion_boundary_is_frame_rate_independent():
    """hold_seconds and the scale window are durations, never sample counts."""
    found = []
    for rate_hz in (5.0, 10.0, 20.0):
        times = np.arange(0, 20, 1.0 / rate_hz)
        rate = np.zeros_like(times)
        rate[times >= 6.0] = 1.0
        rate += np.random.default_rng(4).normal(0, 0.01, len(times))
        found.append(motion_boundary(rate, times, (0.0, 20.0), mode="departs"))

    assert all(f is not None for f in found), f"a sample rate found nothing: {found}"
    assert max(found) - min(found) <= 0.3, f"answers drift with frame rate: {found}"


def test_an_inverted_bracket_is_an_error_not_a_none():
    times = np.arange(0, 20, 0.1)
    with pytest.raises(ValueError, match="ordering upstream is wrong"):
        motion_boundary(np.zeros_like(times), times, (11.8, 10.8))
