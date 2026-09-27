"""Tests for `shapes.py`: reading the local trend of a signal at every sample.

Known answers first (a signal built with a corner where the shape is known), then
the two properties that let the reading travel to footage it has never seen: it
does not care about the signal's scale, and its unreadable edge is a duration, so
it does not care about the frame rate either.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.shapes import SHAPES, read, shape_at, side_changes

SIDE, FLAT, STEEPER, MIN_SIDE = 2.0, 0.08, 1.5, 0.4


def _ramp(t, knots):
    ks, vs = zip(*knots, strict=True)
    return np.interp(t, ks, vs)


def _read(values, t):
    return read(values, t, side=SIDE, flat=FLAT, steeper=STEEPER, min_side=MIN_SIDE)


@pytest.mark.parametrize(
    ("knots", "expected"),
    [
        ([(0, 1), (10, 0), (20, 0)], "drop ends -> flat"),
        ([(0, 0), (10, 1), (20, 1)], "rise ends -> flat"),
        ([(0, 0), (10, 0), (20, 1)], "flat -> starts rising"),
        ([(0, 1), (10, 1), (20, 0)], "flat -> starts falling"),
        ([(0, 0), (10, 1), (20, 0)], "peak"),
        ([(0, 1), (10, 0), (20, 1)], "dip"),
        ([(0, 0), (20, 1)], "keeps rising"),
        ([(0, 1), (20, 0)], "keeps falling"),
        ([(0, 1), (20, 1)], "flat"),
    ],
)
def test_the_shape_at_a_known_corner(knots, expected):
    t = np.arange(0, 20.01, 0.2)
    reading = _read(_ramp(t, knots), t)
    assert reading.shape[int(np.argmin(abs(t - 10)))] == expected


def test_every_pair_of_sides_has_a_name():
    words = ("rising", "falling", "flat")
    assert {(b, a) for b in words for a in words} == set(SHAPES)


def test_speeds_up_and_slows_down():
    """Changes are fractions of the spread over SIDE seconds; 1.5x is the cut."""
    assert shape_at(0.2, 0.5, FLAT, STEEPER) == ("keeps rising", "speeds up")
    assert shape_at(0.5, 0.2, FLAT, STEEPER) == ("keeps rising", "slows down")
    assert shape_at(-0.2, -0.5, FLAT, STEEPER) == ("keeps falling", "speeds up")
    assert shape_at(0.2, 0.25, FLAT, STEEPER) == ("keeps rising", "")


def test_a_reading_checks_shape_and_detail():
    # Short enough that the gentle half still moves well past the flat band
    # relative to the whole signal's spread: twice as steep after the corner.
    t = np.arange(0, 8.01, 0.2)
    reading = _read(_ramp(t, [(0, 0), (4, 4), (8, 12)]), t)
    i = int(np.argmin(abs(t - 4)))
    assert reading.is_(i, "keeps rising")
    assert reading.is_(i, "keeps rising", "speeds up")
    assert not reading.is_(i, "keeps rising", "slows down")


def test_the_scale_of_the_signal_does_not_matter():
    """ "Flat" is a fraction of the signal's own spread, so the same motion filmed
    from twice as far -- every length halved -- reads the same everywhere."""
    t = np.arange(0, 40.01, 0.2)
    rng = np.random.default_rng(1)
    values = np.sin(t / 3.0) + 0.05 * rng.normal(size=t.size)
    assert _read(values, t).shape == _read(values * 0.5, t).shape
    assert _read(values, t).shape == _read(values * 20.0 + 3.0, t).shape


def test_unreadable_only_at_the_edges():
    t = np.arange(0, 20.01, 0.2)
    reading = _read(_ramp(t, [(0, 0), (20, 1)]), t)
    known = [s is not None for s in reading.shape]
    assert not known[0] and not known[-1]
    assert all(known[3:-3])


@pytest.mark.parametrize("rate", [5.0, 10.0, 25.0, 30.0])
def test_the_unreadable_edge_is_a_duration_not_a_sample_count(rate):
    """A side needs MIN_SIDE seconds of data, at any rate. When this was a sample
    count, the unreadable edge was 0.6 s wide at 5 Hz and 0.12 s at 25 Hz, and the
    cycle count on the same synthetic clip changed with the frame rate."""
    t = np.arange(0, 20.0 + 1e-9, 1.0 / rate)
    reading = _read(_ramp(t, [(0, 0), (20, 1)]), t)
    known = np.array([s is not None for s in reading.shape])
    first, last = t[known][0], t[known][-1]
    assert first == pytest.approx(MIN_SIDE, abs=1e-6)
    assert t[-1] - last == pytest.approx(MIN_SIDE, abs=1e-6)


def test_side_changes_are_fractions_of_the_spread():
    """A straight ramp rising by its whole spread over 20 s moves 1/10 of the
    p95-p5 spread per 2 s -- scaled by 1/0.9, because p95-p5 is 90% of the range."""
    t = np.arange(0, 20.01, 0.2)
    before, after = side_changes(_ramp(t, [(0, 0), (20, 1)]), t, SIDE, MIN_SIDE)
    i = int(np.argmin(abs(t - 10)))
    assert before[i] == pytest.approx(0.1 / 0.9, rel=0.02)
    assert after[i] == pytest.approx(0.1 / 0.9, rel=0.02)


def test_gaps_are_fitted_around_not_through():
    t = np.arange(0, 20.01, 0.2)
    values = _ramp(t, [(0, 0), (10, 1), (20, 0)])
    values[38:42] = np.nan  # a dropout at 7.6-8.2 s, inside the next fit's window
    reading = _read(values, t)
    assert reading.shape[int(np.argmin(abs(t - 7.0)))] == "keeps rising"
    assert reading.shape[int(np.argmin(abs(t - 10.0)))] == "peak"
