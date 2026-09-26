"""Bounding boxes from masks, and the smoothing that makes them differentiable.

Design diagram stage 1.5.

Why boxes at all
----------------
Two earlier attempts measured the arm's geometry directly -- a fitted three-link
chain, then dense optical flow -- and both failed for the same underlying reason:
they tried to recover a 3-D articulated motion from a 2-D projection. The chain's
"rigid" links varied 5.2-8.2x in length; the flow-derived house rotation
correlated with the real slew at r = 0.025, because a machine slewing about a
vertical axis *translates* in projection rather than rotating in the image plane.

A bounding box asks a much smaller question -- where is this object, and how wide
and tall does it look -- and every phase cue the design needs turns out to be
answerable from that. The box is a weaker measurement that happens to be strong
enough, which is usually the better trade.

Smoothing, and the one number that matters
------------------------------------------
Box edges jitter by a pixel or two per frame as the segmentation boundary moves.
Differentiating that directly amplifies the jitter, so the centres are smoothed
first with a moving average.

**Window length is the parameter that decides whether any of this works.**
Measured on the development video, sweeping the window and counting how many of
the four phase transitions land inside the +/-0.6 s tolerance:

    0.3 s   4/4
    0.5 s   4/4
    0.9 s   1/4      <- the signal's own features are being averaged away
    1.5 s   1/4

The design diagram specifies ``mean of [i, i+10] applied to [i]``, which at 10 Hz
is a 1.0 s window -- in the collapsed region. The mechanism it describes is right;
the length is not. The default here is 0.5 s, and both are configurable.

Alignment
---------
The diagram's window is **leading**: it averages samples *ahead* of ``i``. That is
as legitimate as any other choice here, because of a result worth knowing:

    mode      |    T1     T2     T3     T4  |  dig   haul   dump  swing
    leading   | -0.07  +0.16  +0.15  -0.55  | +0.23  -0.01  -0.70  +0.59
    centred   | +0.13  +0.36  +0.35  -0.35  | +0.23  -0.01  -0.70  +0.59
    trailing  | +0.33  +0.56  +0.55  -0.15  | +0.23  -0.01  -0.70  +0.59

Alignment shifts every transition, but the **durations are identical** -- a
uniform time shift cancels in every difference, and durations are all the task
grades. So alignment is free, and the default is ``trailing`` only because that is
what was used when all four transitions were verified to pass.

Everything here works in **seconds**, never sample counts, so a video at a
different frame rate gets the same amount of smoothing rather than the same
number of samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

Alignment = Literal["trailing", "centred", "leading"]

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass(frozen=True)
class BoxTrack:
    """One object's box over time, smoothed, in pixels.

    Every array is length ``len(times)`` and carries ``nan`` wherever the object
    had no mask in that sample. A gap is left as a gap: it is not interpolated
    here, because the feature layer needs to know the difference between "the
    bucket did not move" and "the bucket was not found".
    """

    times: np.ndarray  # seconds, from the decoder -- not index / fps
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    # Which samples had a mask BEFORE smoothing. Kept explicitly because the
    # moving average skips NaN, so a smoothed edge is finite wherever any
    # neighbour was found -- deriving this from the smoothed array would report
    # a gap as measured and let the renderer draw a box that was never seen.
    found: np.ndarray

    @property
    def centre_x(self) -> np.ndarray:
        return (self.x0 + self.x1) / 2.0

    @property
    def centre_y(self) -> np.ndarray:
        return (self.y0 + self.y1) / 2.0

    @property
    def width(self) -> np.ndarray:
        """Inclusive pixel width: a one-pixel-wide box has width 1, not 0."""
        return self.x1 - self.x0 + 1.0

    @property
    def height(self) -> np.ndarray:
        return self.y1 - self.y0 + 1.0

    @property
    def aspect_ratio(self) -> np.ndarray:
        """Width over height. Carries the bucket's tipping: it is widest seen
        side-on and narrows as it rotates over to empty."""
        return self.width / np.maximum(self.height, np.finfo(float).eps)

    def __len__(self) -> int:
        return len(self.times)


def box_of(mask: np.ndarray) -> Box | None:
    """The tightest axis-aligned box containing every true pixel.

    ``None`` when the mask is empty, which is a normal outcome -- the bucket
    leaves the frame or the tracker loses it -- and not an error.
    """
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def raw_boxes(
    masks_by_sample: dict[int, np.ndarray] | list[np.ndarray | None],
    count: int,
) -> np.ndarray:
    """Un-smoothed boxes as an ``(count, 4)`` array, ``nan`` where absent."""
    out = np.full((count, 4), np.nan)
    getter = (
        masks_by_sample.get
        if isinstance(masks_by_sample, dict)
        else lambda i: masks_by_sample[i] if i < len(masks_by_sample) else None
    )
    for index in range(count):
        found = box_of(getter(index))
        if found is not None:
            out[index] = found
    return out


def moving_average(
    values: np.ndarray,
    times: np.ndarray,
    window_seconds: float,
    alignment: Alignment = "trailing",
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """Average ``values`` over a window of real time.

    The window is expressed in seconds and converted through the *observed*
    sample spacing, so the same call smooths a 25 fps video and a 30 fps video by
    the same physical amount.

    ``nan`` inputs are skipped rather than propagated: a sample whose window
    contains enough valid neighbours gets their mean.

    ``min_valid_fraction`` is why that is bounded. Averaging whatever survives
    is an unbiased estimate of the signal at the *surviving samples' centroid*,
    not at the window's centre -- so a window with one sample left is not a
    smoothed value at this instant, it is an unsmoothed value from up to
    ``(width - 1) * dt`` ago, with the variance to match. At the default half,
    the worst time shift is a quarter of the window rather than all of it.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {values.shape}")
    if window_seconds <= 0:
        return values.copy()

    width = _window_samples(window_seconds, times)
    if width <= 1:
        return values.copy()

    needed = max(1, int(np.ceil(width * min_valid_fraction)))
    out = np.full(len(values), np.nan)
    for index in range(len(values)):
        lo, hi = _window_bounds(index, width, alignment, len(values))
        segment = values[lo:hi]
        segment = segment[np.isfinite(segment)]
        # A truncated window at the clip's edge is short but not missing data,
        # so it is judged against what it could contain, not against `width`.
        if segment.size >= min(needed, hi - lo):
            out[index] = segment.mean()
    return out


def smooth_boxes(
    raw: np.ndarray,
    times: np.ndarray,
    window_seconds: float,
    alignment: Alignment = "trailing",
) -> BoxTrack:
    """Smooth each box edge independently and return the track.

    Edges are smoothed rather than centres-and-sizes because the features need
    both: the centre carries position, and width and height carry the bucket's
    orientation through the aspect ratio. Smoothing the four edges gives a
    consistent answer for all of them.
    """
    if raw.shape[1] != 4:
        raise ValueError(f"expected (n, 4) boxes, got {raw.shape}")
    if len(raw) != len(times):
        raise ValueError(f"{len(raw)} boxes but {len(times)} timestamps")

    found = np.isfinite(raw[:, 0])  # from the RAW boxes, before any smoothing
    edges = [moving_average(raw[:, i], times, window_seconds, alignment) for i in range(4)]
    track = BoxTrack(np.asarray(times, dtype=float), *edges, found=found)

    missing = int((~found).sum())
    if missing:
        smoothed_over = int((np.isfinite(track.x0) & ~found).sum())
        log.info(
            "box track: %d of %d samples had no mask (%d of them filled by smoothing)",
            missing,
            len(track),
            smoothed_over,
        )
    return track


def _window_samples(window_seconds: float, times: np.ndarray) -> int:
    """How many samples span ``window_seconds``, from the real timestamps.

    The median spacing is used rather than the mean so one long gap -- a dropped
    frame, a decoder hiccup -- does not stretch the window for the whole video.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return 1
    spacing = float(np.median(np.diff(times)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"timestamps are not increasing (median spacing {spacing})")
    return max(1, round(window_seconds / spacing))


def _window_bounds(
    index: int, width: int, alignment: Alignment, total: int
) -> tuple[int, int]:
    """Half-open slice bounds for the window at ``index``."""
    if alignment == "trailing":  # the w samples ENDING at index
        return max(0, index - width + 1), index + 1
    if alignment == "leading":  # the w samples STARTING at index -- the diagram's
        return index, min(total, index + width)
    if alignment == "centred":
        # (width - 1) // 2, not width // 2: the latter yields width + 1 samples
        # whenever width is even, so "centred" would smooth harder than the other
        # two and the claim that alignment is free would quietly stop holding.
        half = (width - 1) // 2
        return max(0, index - half), min(total, index - half + width)
    raise ValueError(f"unknown alignment {alignment!r}")
