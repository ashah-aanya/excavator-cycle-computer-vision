"""Tests for rates and noise: how fast a signal changes, and how much it jitters.

Two things these tests protect
------------------------------
A **rate**'s noise must be measured from the rate itself, by differencing, not read off
its magnitude: a plateau at any level, a steady ramp and a slow drift all cancel out,
leaving the sample-to-sample jitter.

And "not measured" must never read as "not moving". A tracking gap at either end of a
clip stays NaN through the derivative; otherwise a search for rest would land on the
flat run the gap left behind and report the moment the tracker acquired the bucket.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.rates import derivative, noise_scale

# --- rest is zero, for a rate --------------------------------------------


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


# --- regressions from the physics review ----------------------------------


def test_a_leading_gap_does_not_become_a_fake_rest_plateau():
    """np.interp clamps outside its range, holding the first finite value flat.
    For a RATE that manufactures the most convincing possible rest -- and
    rest is "near zero" -- so a search for rest would
    land on it and report the moment the tracker acquired the bucket."""
    height = np.concatenate([np.full(10, np.nan), np.linspace(0.1, 0.5, 30)])
    times = np.arange(40) * 0.1
    rate = derivative(height, times)
    assert np.isnan(rate[:10]).all(), "unmeasured must not read as zero"
    assert np.isfinite(rate[12:]).all()


def test_a_trailing_gap_stays_nan_too():
    height = np.concatenate([np.linspace(0.1, 0.5, 30), np.full(10, np.nan)])
    assert np.isnan(derivative(height, np.arange(40) * 0.1)[-10:]).all()


def test_interior_gaps_are_still_bridged():
    """Only the ends are refused; a short dropout mid-clip is interpolated so
    the filter can run across it."""
    height = np.linspace(0.1, 0.5, 40).copy()
    height[18:21] = np.nan
    rate = derivative(height, np.arange(40) * 0.1)
    assert np.isfinite(rate[18:21]).all()


def test_a_broken_time_base_raises_instead_of_scaling_every_rate():
    """A silent fallback to dt = 1.0 s scales every derivative by 1/dt -- 10x at
    10 Hz -- and the result stays plausible, so nothing downstream notices."""
    with pytest.raises(ValueError, match="not increasing"):
        derivative(np.arange(10.0), np.zeros(10))


def test_polyorder_is_honoured_and_checked():
    """The polynomial order is a real setting: a cubic fit of a cubic recovers its slope
    exactly where a quadratic cannot, and an order the window cannot support is refused
    with a message, not a scipy traceback."""
    times = np.arange(60) * 0.1
    signal = (times - 3.0) ** 3
    truth = 3.0 * (times - 3.0) ** 2
    inside = slice(10, 50)
    quadratic = derivative(signal, times, window_seconds=0.9, polyorder=2)
    cubic = derivative(signal, times, window_seconds=0.9, polyorder=3)
    assert np.abs(cubic[inside] - truth[inside]).max() < 1e-6
    assert np.abs(quadratic[inside] - truth[inside]).max() > 1e-3
    with pytest.raises(ValueError, match="polyorder"):
        derivative(signal, times, window_seconds=0.2, polyorder=3)
