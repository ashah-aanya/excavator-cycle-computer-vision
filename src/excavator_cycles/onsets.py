"""Finding the moment a signal leaves rest, or settles into it.

Why a departure and not a threshold crossing
--------------------------------------------
A threshold scaled off a signal's peak is **systematically late** on anything
that accelerates from rest, and late by a different amount for each transition
-- which is the differential bias a +/-0.6 s tolerance cannot absorb. Measured
on the development video's swing onset:

    parked            0.02 - 0.11
    true onset        0.23          <- the machine has started moving
    threshold trips   1.45          <- 2.9 s later

Same signal, same clip. The signal was never the problem.

So: find the obvious excursion, then walk **back** to where the signal departed
its resting level. Pass 1 may be late for free, because pass 1's answer is a
bracket rather than a time.

The four things that went wrong in the first attempt
----------------------------------------------------
Recorded because each one is easy to write again:

1. **The baseline was estimated on the ramp.** Taking "the few seconds before
   the peak" as rest fails when the signal has been climbing through them: the
   baseline comes out too high AND its spread too wide, so the band is enormous
   and the walk-back stops beside the peak. Rest is instead estimated from the
   **lowest-spread window** anywhere in the bracket, which is a definition
   rather than an assumption about where the event is.
2. **The walk stopped at the first quiet sample.** Real signals cross a band
   repeatedly; a single dip inside it ended the search. It now requires the
   signal to stay inside for a sustained stretch.
3. **The bracket selected the wrong occurrence.** Taking "the first time the
   gate opens" caught the tail of the *previous* cycle. `strongest_interval`
   ranks by duration times magnitude instead.
4. **Hand-written constants.** `-0.02`, `0.012`, "the 25th percentile" -- all
   fitted to one clip, which this project may not ship.

Units
-----
Every parameter here is either **dimensionless** (a multiple of the signal's own
MAD) or **a duration in seconds** (converted through the sample spacing). No
pixel counts, no per-video magnitudes. A signal measured on footage shot twice
as far, or sampled at a different rate, is treated identically -- the thresholds
come from the signal's own statistics, so they adapt on their own.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _spacing(times: np.ndarray) -> float:
    """Seconds per sample, from the data rather than from a constant."""
    if len(times) < 2:
        return 1.0
    step = float(np.median(np.diff(times)))
    return step if step > 0 else 1.0


def _samples(seconds: float, times: np.ndarray) -> int:
    return max(1, round(seconds / _spacing(times)))


def rest_level(
    signal: np.ndarray,
    times: np.ndarray,
    window_seconds: float = 0.8,
    quiet_percentile: float = 20.0,
) -> tuple[float, float]:
    """Estimate what this signal looks like when nothing is happening.

    Rest is the **lowest-spread window** in the series, not the stretch before
    some event -- estimating a baseline from just-before-the-peak requires
    already knowing where the peak begins, which is the thing being searched
    for.

    Returns ``(level, scale)``, where ``scale`` is a robust standard deviation
    (MAD x 1.4826). Both are NaN if the signal has no finite values.

    ``window_seconds`` is a statement about the machine -- an excavator holds
    still for at least a moment between motions -- not about any one video, and
    it is converted to samples through the observed sample spacing.
    """
    finite = np.isfinite(signal)
    if not finite.any():
        return float("nan"), float("nan")

    width = min(_samples(window_seconds, times), int(finite.sum()))
    if width < 3:
        values = signal[finite]
        level = float(np.median(values))
        return level, float(np.median(np.abs(values - level)) * 1.4826)

    levels, scales = [], []
    for start in range(0, len(signal) - width + 1):
        chunk = signal[start : start + width]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk) < 3:
            continue
        level = float(np.median(chunk))
        levels.append(level)
        scales.append(float(np.median(np.abs(chunk - level)) * 1.4826))

    if not scales:
        values = signal[finite]
        level = float(np.median(values))
        return level, float(np.median(np.abs(values - level)) * 1.4826)

    # The MINIMUM spread across many windows is biased low -- over hundreds of
    # windows it finds the luckiest quiet patch rather than the noise floor, so
    # the band comes out narrower than the noise and ordinary noise reads as an
    # excursion. A low percentile of the window spreads is quiet without being
    # lucky. Measured: using the minimum made the detector fire on pure noise.
    scales = np.asarray(scales)
    levels = np.asarray(levels)
    cut = float(np.percentile(scales, quiet_percentile))
    quietest = scales <= max(cut, np.finfo(float).tiny)
    return float(np.median(levels[quietest])), float(np.median(scales[quietest]))


def strongest_interval(
    signal: np.ndarray,
    times: np.ndarray,
    open_fraction: float = 0.25,
    min_seconds: float = 0.5,
) -> tuple[float, float] | None:
    """The most convincing stretch where a gating signal is raised.

    Ranked by duration times mean height, so a brief spike -- the tail of a
    previous cycle, say -- cannot outrank the real occurrence. Taking the
    *first* raised stretch instead is what made the dumping bracket open on the
    previous cycle's dump and search a 21-second window.

    The level counts as raised above ``open_fraction`` of the signal's own p95,
    which is the project's threshold convention: a fraction fixed across all
    videos, applied to a magnitude measured within each one.
    """
    finite = np.isfinite(signal)
    if not finite.any():
        return None

    peak = float(np.nanpercentile(signal, 95))
    if not np.isfinite(peak) or peak <= 0:
        return None
    raised = finite & (signal >= open_fraction * peak)
    if not raised.any():
        return None

    least = _samples(min_seconds, times)
    best, best_merit = None, -np.inf
    start = None
    for index in range(len(raised) + 1):
        inside = index < len(raised) and raised[index]
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            if index - start >= least:
                merit = (index - start) * float(np.nanmean(signal[start:index]))
                if merit > best_merit:
                    best, best_merit = (start, index - 1), merit
            start = None
    if best is None:
        return None
    return float(times[best[0]]), float(times[best[1]])


def rest_boundary(
    signal: np.ndarray,
    times: np.ndarray,
    bracket: tuple[float, float] | None = None,
    mode: str = "departs",
    sigma: float = 3.0,
    hold_seconds: float = 0.3,
    rest_window_seconds: float = 0.8,
) -> float | None:
    """When does ``signal`` leave rest (``departs``) or settle into it (``arrives``)?

    Args:
        signal: one value per sample; NaNs are tolerated.
        times: seconds per sample, parallel to ``signal``.
        bracket: optional ``(start, end)`` in seconds to search within. Pass 1's
            job. Without one the whole series is searched.
        mode: ``"departs"`` walks backward from the excursion to where the
            signal left rest -- T2, T3, T4. ``"arrives"`` walks forward to where
            it settled -- T1, the bucket reaching the material.
        sigma: band half-width, in multiples of rest's robust deviation.
            Dimensionless, so it travels between videos and between signals.
        hold_seconds: the signal must stay inside the band this long for the
            walk to stop. One sample dipping back across a band is noise, not
            an arrival; without this the search halts on the first flicker.
        rest_window_seconds: how long a stretch is used to estimate rest.

    Returns the time, or None when no excursion leaves the band at all.
    """
    if mode not in ("departs", "arrives"):
        raise ValueError(f"mode must be 'departs' or 'arrives', not {mode!r}")

    times = np.asarray(times, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) != len(times):
        raise ValueError(f"{len(signal)} values but {len(times)} times; they must be parallel")

    inside_bracket = np.ones(len(signal), dtype=bool)
    if bracket is not None:
        # An inverted bracket means the caller's ordering logic went wrong --
        # a later transition was placed before an earlier one. Silently
        # returning None hides that as "no event here", which is a different
        # and much more believable failure. It has already happened once.
        if bracket[1] <= bracket[0]:
            raise ValueError(
                f"bracket ({bracket[0]:.3f}, {bracket[1]:.3f}) ends at or before it "
                "starts; the transition ordering upstream is wrong"
            )
        inside_bracket = (times >= bracket[0]) & (times <= bracket[1])
    window = np.where(inside_bracket & np.isfinite(signal))[0]
    if len(window) < 3:
        return None

    level, scale = rest_level(signal[window], times[window], rest_window_seconds)
    if not np.isfinite(level) or not np.isfinite(scale) or scale <= 0:
        return None

    band = sigma * scale
    hold = _samples(hold_seconds, times)

    # The excursion must be SUSTAINED, not one extreme sample. Pure noise always
    # produces a sample beyond 3 sigma, and a single-frame spike is not an event;
    # without this the detector fires on both, which it did.
    outside = np.abs(signal[window] - level) > band
    runs, start = [], None
    for position in range(len(outside) + 1):
        marked = position < len(outside) and outside[position]
        if marked and start is None:
            start = position
        elif not marked and start is not None:
            if position - start >= hold:
                runs.append((start, position))
            start = None
    if not runs:
        return None  # nothing ever leaves rest; say so rather than guess

    loudest = max(
        runs, key=lambda r: float(np.nanmax(np.abs(signal[window[r[0] : r[1]]] - level)))
    )
    span = window[loudest[0] : loudest[1]]
    peak = span[int(np.argmax(np.abs(signal[span] - level)))]
    step = -1 if mode == "departs" else 1
    limit = window[0] if mode == "departs" else window[-1]

    index = peak
    quiet = 0
    while index != limit:
        nxt = index + step
        value = signal[nxt]
        quiet = quiet + 1 if np.isfinite(value) and abs(value - level) <= band else 0
        if quiet >= hold:
            # `index` walked `hold` samples into the quiet stretch; the boundary
            # is where it first went quiet, not where we stopped looking.
            return float(times[index + (hold - 1) * -step])
        index = nxt

    return float(times[limit])


def smooth(signal: np.ndarray, times: np.ndarray, window_seconds: float = 0.9) -> np.ndarray:
    """Zero-phase smoothing, with the window given in SECONDS.

    Savitzky-Golay is symmetric, so it does not move a departure or an extremum
    in time. A causal filter would lag every boundary by the same amount, which
    is the systematic error a +/-0.6 s tolerance cannot absorb.

    The window must be expressed in seconds and converted through the observed
    sample spacing. Giving it in *samples* -- which is what ``savgol_filter``
    takes directly -- makes the amount of smoothing depend on the frame rate: 9
    samples is 0.9 s at 10 Hz and 0.45 s at 20 Hz, so the same footage sampled
    differently would yield different onsets. ``features.py`` already does this
    correctly; a scratch script that did not is what prompted this helper.
    """
    from scipy.signal import savgol_filter

    filled = _interpolate(signal)
    width = _odd(max(3, _samples(window_seconds, times)))
    if len(filled) <= width or not np.isfinite(filled).any():
        return filled
    return savgol_filter(filled, width, 2)


def derivative(
    signal: np.ndarray, times: np.ndarray, window_seconds: float = 0.9
) -> np.ndarray:
    """Zero-phase first derivative, window in seconds. See ``smooth``."""
    from scipy.signal import savgol_filter

    filled = _interpolate(signal)
    width = _odd(max(3, _samples(window_seconds, times)))
    if len(filled) <= width or not np.isfinite(filled).any():
        return np.full_like(filled, np.nan)
    return savgol_filter(filled, width, 2, deriv=1, delta=_spacing(times))


def _interpolate(values: np.ndarray) -> np.ndarray:
    """Fill interior gaps so a filter can run; callers keep their own validity."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.all() or finite.sum() < 2:
        return values.copy()
    index = np.arange(len(values))
    return np.interp(index, index[finite], values[finite])


def _odd(value: int) -> int:
    return value if value % 2 else value + 1
