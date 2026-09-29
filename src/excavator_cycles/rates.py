"""Rates and noise: how fast is a signal changing, and how much does it jitter?

Every parameter here is either **dimensionless** (a multiple of the signal's own MAD) or
**a duration in seconds** (converted through the sample spacing). No pixel counts, no
per-video magnitudes: a signal measured on footage shot twice as far, or sampled at a
different rate, is treated identically.
"""

from __future__ import annotations

import numpy as np


def _spacing(times: np.ndarray) -> float:
    """Median sample interval, in seconds.

    Raises rather than substituting a default. A silent fallback to 1.0 s would
    scale every derivative by 1/dt -- 10x at 10 Hz -- and the result stays
    plausible, so nothing downstream would notice. ``boxes._window_samples``
    already refuses the same input; the two must agree.
    """
    if len(times) < 2:
        raise ValueError(f"need at least 2 timestamps to find a spacing, got {len(times)}")
    step = float(np.median(np.diff(times)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"timestamps are not increasing (median spacing {step})")
    return step


def _samples(seconds: float, times: np.ndarray) -> int:
    return max(1, round(seconds / _spacing(times)))


def noise_scale(signal: np.ndarray) -> float:
    """How big is this signal's noise? MAD of successive differences / sqrt(2).

    Differencing cancels any trend -- a plateau at any level, a steady ramp, a
    slow drift -- and leaves the sample-to-sample noise. The ``sqrt(2)`` undoes
    the variance doubling that comes from subtracting two noisy samples. MAD
    rather than a standard deviation so that one genuine jump (a step edge
    contributes exactly one large difference) cannot inflate the estimate.

    Why not "the spread of the quietest window", which this module used first
    ------------------------------------------------------------------------
    Because a window short enough to be quiet is too short to measure a spread
    with, and taking a low percentile across many noisy window estimates lands
    in their low tail. Measured against a known ``sigma = 0.0100``:

        window   median-of-bottom-20%   the p20 itself
        0.8 s        0.46x                  0.61x
        1.5 s        0.65x                  0.76x
        2.5 s        0.77x                  0.80x

    A nominal 3-sigma band was really 1.37 sigma, which ordinary noise clears
    17% of the time -- so noise read as motion and every walk stopped early.
    Widening the window narrows the bias but never removes it, and a longer
    window is less likely to be quiet at all.

    This estimator instead returns 0.85-0.86x on a step, on a ramp, and on a
    signal that is moving for 90% of its length. The bias is small and, more
    importantly, **stable**: it does not depend on how much of the clip is
    moving, which is the property that matters when the same code has to serve
    a digging plateau and a full-speed slew.
    """
    values = np.asarray(signal, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    steps = np.diff(values)
    return float(np.median(np.abs(steps - np.median(steps))) * 1.4826 / np.sqrt(2.0))


def derivative(
    signal: np.ndarray, times: np.ndarray, window_seconds: float = 0.9
) -> np.ndarray:
    """Zero-phase first derivative, window in seconds.

    Savitzky-Golay: fit a low-order polynomial over a sliding window and report the
    FITTED derivative, so the result is smooth and has no phase lag. `delta` converts
    it from per-sample to per-second, which is what makes the window a duration
    rather than a frame count.
    """
    from scipy.signal import savgol_filter

    filled = _interpolate(signal)
    width = _odd(max(3, _samples(window_seconds, times)))
    if len(filled) <= width or not np.isfinite(filled).any():
        return np.full_like(filled, np.nan)
    out = savgol_filter(np.nan_to_num(filled), width, 2, deriv=1, delta=_spacing(times))
    return _restore_gaps(out, filled)


def _interpolate(values: np.ndarray) -> np.ndarray:
    """Fill INTERIOR gaps so a filter can run. Never extrapolate past the ends.

    ``np.interp`` clamps outside the range it was given, holding the first and
    last finite values flat. For a rate that is disastrous rather than merely
    inaccurate: rest is "near zero", so a clip whose tracking fails at either end
    is handed a perfectly flat run of samples -- the most convincing possible rest
    plateau -- and a search for rest will happily land on it and report a start
    that is really the moment the tracker acquired the bucket.

    Leading and trailing gaps therefore stay NaN, and the filters below return
    NaN there too, so "not measured" cannot be mistaken for "not moving".
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.all() or finite.sum() < 2:
        return values.copy()
    index = np.arange(len(values))
    filled = np.interp(index, index[finite], values[finite])
    first, last = int(index[finite][0]), int(index[finite][-1])
    filled[:first] = np.nan
    filled[last + 1 :] = np.nan
    return filled


def _restore_gaps(filtered: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Put the unmeasured span back as NaN after filtering.

    ``savgol_filter`` cannot take NaN, so the ends are zero-filled to run it and
    then blanked again here. Without this the zeros would read as a rate of
    exactly zero -- see ``_interpolate``.
    """
    out = np.asarray(filtered, dtype=np.float64).copy()
    out[~np.isfinite(source)] = np.nan
    return out


def _odd(value: int) -> int:
    return value if value % 2 else value + 1
