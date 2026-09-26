"""Boxes and their smoothing (design diagram stage 1.5).

The property that matters most here is that the window is a duration, not a
sample count: the same call must smooth a 25 fps and a 30 fps video by the same
amount of *time*. Several tests below exist only to pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.boxes import (
    BoxTrack,
    box_of,
    moving_average,
    raw_boxes,
    smooth_boxes,
)


def _mask(x0, y0, x1, y1, shape=(40, 60)):
    m = np.zeros(shape, dtype=bool)
    m[y0 : y1 + 1, x0 : x1 + 1] = True
    return m


# --- box_of ---------------------------------------------------------------


def test_box_of_is_tight_and_inclusive():
    assert box_of(_mask(10, 5, 20, 15)) == (10.0, 5.0, 20.0, 15.0)


def test_box_of_a_single_pixel():
    assert box_of(_mask(7, 7, 7, 7)) == (7.0, 7.0, 7.0, 7.0)


def test_box_of_ignores_shape_and_finds_only_true_pixels():
    m = _mask(3, 3, 5, 5)
    m[30, 50] = True  # one stray pixel far away
    assert box_of(m) == (3.0, 3.0, 50.0, 30.0), "the box must contain every true pixel"


def test_box_of_an_empty_mask_is_none_not_an_error():
    """The tracker losing the bucket is normal, not exceptional."""
    assert box_of(np.zeros((10, 10), dtype=bool)) is None
    assert box_of(None) is None


def test_raw_boxes_leaves_gaps_as_nan():
    raw = raw_boxes({0: _mask(1, 1, 3, 3), 2: _mask(4, 4, 6, 6)}, count=4)
    assert raw.shape == (4, 4)
    assert np.allclose(raw[0], [1, 1, 3, 3])
    assert np.isnan(raw[1]).all(), "sample 1 had no mask"
    assert np.isnan(raw[3]).all()


# --- the window is a duration ---------------------------------------------


def test_window_is_seconds_not_samples():
    """The same 1 s window must span more samples when sampling is denser."""
    v = np.arange(20, dtype=float)
    dense = moving_average(v, np.arange(20) * 0.1, 1.0, "trailing")  # 10 samples
    sparse = moving_average(v, np.arange(20) * 0.5, 1.0, "trailing")  # 2 samples
    # at index 19: dense averages 10..19 (=14.5), sparse averages 18..19 (=18.5)
    assert dense[19] == pytest.approx(14.5)
    assert sparse[19] == pytest.approx(18.5)


def test_median_spacing_survives_one_dropped_sample():
    """A single gap must not stretch the window for the whole video."""
    t = np.arange(20) * 0.1
    t[10:] += 2.0  # a two-second hole in the middle
    v = np.arange(20, dtype=float)
    assert moving_average(v, t, 0.5, "trailing")[5] == pytest.approx(
        moving_average(v, np.arange(20) * 0.1, 0.5, "trailing")[5]
    )


def test_non_increasing_timestamps_raise_rather_than_silently_misbehave():
    with pytest.raises(ValueError, match="not increasing"):
        moving_average(np.arange(5.0), np.zeros(5), 0.5)


# --- alignment ------------------------------------------------------------


def test_trailing_leading_and_centred_pick_the_windows_they_say():
    v = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    t = np.arange(5) * 0.1
    # a 0.3s window is 3 samples
    assert moving_average(v, t, 0.3, "trailing")[3] == pytest.approx(20.0)  # 10,20,30
    assert moving_average(v, t, 0.3, "leading")[1] == pytest.approx(20.0)  # 10,20,30
    assert moving_average(v, t, 0.3, "centred")[2] == pytest.approx(20.0)  # 10,20,30


def test_alignment_shifts_a_ramp_but_not_its_slope():
    """Why alignment is free: it moves events in time, identically for all of
    them, so it cancels in every duration."""
    t = np.arange(40) * 0.1
    v = np.where(t < 2.0, 0.0, (t - 2.0) * 5.0)
    slopes = {}
    for mode in ("trailing", "centred", "leading"):
        s = moving_average(v, t, 0.5, mode)
        slopes[mode] = np.nanmax(np.diff(s))
    assert slopes["trailing"] == pytest.approx(slopes["centred"], rel=1e-9)
    assert slopes["centred"] == pytest.approx(slopes["leading"], rel=1e-9)


def test_unknown_alignment_is_rejected():
    with pytest.raises(ValueError, match="unknown alignment"):
        moving_average(np.arange(5.0), np.arange(5) * 0.1, 0.3, "sideways")


# --- gaps -----------------------------------------------------------------


def test_nan_is_skipped_not_propagated():
    v = np.array([1.0, np.nan, 3.0, 5.0])
    out = moving_average(v, np.arange(4) * 0.1, 0.3, "trailing")
    assert out[2] == pytest.approx(2.0), "mean of 1 and 3, skipping the nan"


def test_a_window_with_no_data_at_all_stays_nan():
    v = np.array([np.nan, np.nan, np.nan, 4.0])
    assert np.isnan(moving_average(v, np.arange(4) * 0.1, 0.2, "trailing")[1])


def test_zero_window_is_a_passthrough():
    v = np.array([1.0, 9.0, 2.0])
    assert np.allclose(moving_average(v, np.arange(3) * 0.1, 0.0), v)


# --- BoxTrack -------------------------------------------------------------


def _track(n=10):
    raw = np.stack(
        [np.full(n, 10.0), np.full(n, 20.0), np.full(n, 30.0), np.full(n, 26.0)], axis=1
    )
    return smooth_boxes(raw, np.arange(n) * 0.1, 0.3, "trailing")


def test_derived_geometry_is_inclusive_and_consistent():
    t = _track()
    assert t.centre_x[-1] == pytest.approx(20.0)
    assert t.centre_y[-1] == pytest.approx(23.0)
    assert t.width[-1] == pytest.approx(21.0), "30-10+1: inclusive"
    assert t.height[-1] == pytest.approx(7.0)
    assert t.aspect_ratio[-1] == pytest.approx(3.0)


def test_aspect_ratio_survives_a_zero_height_box():
    raw = np.array([[5.0, 5.0, 9.0, 5.0]])  # a one-pixel-tall box
    t = smooth_boxes(raw, np.array([0.0]), 0.0)
    assert np.isfinite(t.aspect_ratio[0])


def test_found_reports_which_samples_had_a_mask():
    raw = np.full((4, 4), np.nan)
    raw[1] = [1.0, 1.0, 2.0, 2.0]
    t = smooth_boxes(raw, np.arange(4) * 0.1, 0.0)
    assert list(t.found) == [False, True, False, False]


def test_smooth_boxes_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="timestamps"):
        smooth_boxes(np.zeros((5, 4)), np.arange(3) * 0.1, 0.3)


def test_smooth_boxes_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        smooth_boxes(np.zeros((5, 3)), np.arange(5) * 0.1, 0.3)


def test_boxtrack_length_is_the_sample_count():
    assert len(_track(n=7)) == 7
    assert isinstance(_track(), BoxTrack)


# --- regressions from the physics review ----------------------------------


def test_found_is_measured_before_smoothing_not_after():
    """The moving average skips NaN, so a smoothed edge is finite wherever any
    neighbour was found. Deriving `found` from that would report a gap as
    measured, and the renderer would draw a box that was never seen."""
    raw = np.tile([10.0, 20.0, 30.0, 26.0], (12, 1))
    raw[4:8] = np.nan  # genuinely absent for four samples
    track = smooth_boxes(raw, np.arange(12) * 0.1, 0.5, "trailing")
    assert list(track.found[4:8]) == [False] * 4, "found must reflect the raw boxes"
    assert int((~track.found).sum()) == 4


def test_a_window_with_too_little_data_declines_to_answer():
    """One surviving sample is not a smoothed value at this instant; it is an
    unsmoothed value from up to (width-1)*dt ago, with the variance to match."""
    v = np.array([1.0, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0])
    out = moving_average(v, np.arange(8) * 0.1, 0.5, "trailing", min_valid_fraction=0.5)
    assert np.isnan(out[5]), "only one valid sample in a 5-wide window"


def test_a_truncated_window_at_the_start_is_not_treated_as_missing_data():
    """Sample 0 has a one-sample window by construction, not by loss."""
    v = np.arange(8, dtype=float)
    out = moving_average(v, np.arange(8) * 0.1, 0.5, "trailing", min_valid_fraction=0.5)
    assert np.isfinite(out[0])


@pytest.mark.parametrize("width", [4, 5, 6, 7])
def test_every_alignment_uses_the_same_number_of_samples(width):
    """`width // 2` yields width+1 samples when width is even, so `centred`
    would smooth harder than the other two and the claim that alignment is free
    would stop holding."""
    from excavator_cycles.boxes import _window_bounds

    spans = {
        mode: (lambda b: b[1] - b[0])(_window_bounds(20, width, mode, 100))
        for mode in ("trailing", "centred", "leading")
    }
    assert set(spans.values()) == {width}, spans
