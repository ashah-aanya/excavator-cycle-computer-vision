"""The local shape of a signal at every sample: is it peaking, levelling off, taking off?

A phase starts at a moment, and a moment shows up as a change in a signal's TREND,
not as its level. "Height is low" is true for the whole of a dig; "speed was falling
and has just gone flat" is true only where the swing ends and the dig begins. So
the onset cues in `fsm.py` read shapes, and this module is where a shape is read.

How
---
At every sample, fit a straight line to the ``side`` seconds before it and to the
``side`` seconds after it. Each side is ``rising``, ``falling`` or ``flat``, where
flat means the line moves less than ``flat`` of the signal's spread over ``side``
seconds. The pair names the shape::

    falling, flat     "drop ends -> flat"      rising, falling   "peak"
    rising, flat      "rise ends -> flat"      falling, rising   "dip"
    flat, rising      "flat -> starts rising"  rising, rising    "keeps rising"
    flat, falling     "flat -> starts falling" falling, falling  "keeps falling"
    flat, flat        "flat"

When both sides move the same way, a slope ratio above ``steeper`` adds "speeds up"
(or below its inverse, "slows down").

What is global and what is local
--------------------------------
The one number read from the whole signal is its spread (p95 - p5), which says what
"flat" MEANS on this video: a line that moves 8% of the signal's usual range is
flat whether the machine is near the camera or far away. Everything else is read
from the few seconds around each sample. That is `fsm.py`'s rule -- a global
statistic may say what a word means, never when anything happened.

At the edges of the clip a side has fewer than ``side`` seconds to fit. It is fitted
on what exists, down to ``min_side`` SECONDS (and never fewer than three samples),
so a transition close to either end is not lost merely for being there; below that
the side reads as unknown and the shape is ``None``. The floor is a duration, not a
sample count, on purpose: "three samples" is 0.6 s at 5 Hz and 0.12 s at 25 Hz, and
a count made the unreadable edge -- and with it which cycles were found -- change
with the frame rate.

The evaluation tooling implements the same reading independently, because the
pipeline may not import or name anything on the evaluation side; a test outside
the pipeline holds the two equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPES = {
    ("falling", "flat"): "drop ends -> flat",
    ("rising", "flat"): "rise ends -> flat",
    ("flat", "rising"): "flat -> starts rising",
    ("flat", "falling"): "flat -> starts falling",
    ("rising", "falling"): "peak",
    ("falling", "rising"): "dip",
    ("rising", "rising"): "keeps rising",
    ("falling", "falling"): "keeps falling",
    ("flat", "flat"): "flat",
}

# Fewer samples than this cannot say which way a side is going, however long they span.
_MIN_POINTS = 3


@dataclass(frozen=True)
class Reading:
    """One signal's shape at every sample. ``None`` where it cannot be read."""

    shape: tuple[str | None, ...]
    detail: tuple[str, ...]

    def is_(self, index: int, shape: str, detail: str = "") -> bool:
        return self.shape[index] == shape and (not detail or self.detail[index] == detail)


def _slope(times: np.ndarray, values: np.ndarray, min_span: float) -> float:
    ok = np.isfinite(values)
    if ok.sum() < _MIN_POINTS or times[ok][-1] - times[ok][0] < min_span - 1e-9:
        return float("nan")
    return float(np.polyfit(times[ok], values[ok], 1)[0])


def side_changes(
    values: np.ndarray, times: np.ndarray, side: float, min_side: float
) -> tuple[np.ndarray, np.ndarray]:
    """Line-fit change over ``side`` seconds before and after each sample, as a
    fraction of the signal's p95 - p5 spread."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    step = float(np.median(np.diff(times)))
    n = max(1, round(side / step))
    finite = values[np.isfinite(values)]
    spread = (
        float(np.percentile(finite, 95) - np.percentile(finite, 5)) if finite.size else 0.0
    )
    spread = spread if spread > 0 else 1.0
    before = np.full(len(values), np.nan)
    after = np.full(len(values), np.nan)
    for i in range(len(values)):
        lo, hi = max(0, i - n), min(len(values), i + n + 1)
        before[i] = _slope(times[lo : i + 1], values[lo : i + 1], min_side) * side / spread
        after[i] = _slope(times[i:hi], values[i:hi], min_side) * side / spread
    return before, after


def _word(change: float, flat: float) -> str | None:
    if not np.isfinite(change):
        return None
    if abs(change) < flat:
        return "flat"
    return "rising" if change > 0 else "falling"


def shape_at(
    before: float, after: float, flat: float, steeper: float
) -> tuple[str | None, str]:
    """(shape, detail) from one sample's two side changes."""
    b, a = _word(before, flat), _word(after, flat)
    if b is None or a is None:
        return None, ""
    detail = ""
    if b == a and b != "flat":
        ratio = abs(after) / abs(before)
        if ratio > steeper:
            detail = "speeds up"
        elif ratio < 1 / steeper:
            detail = "slows down"
    return SHAPES[(b, a)], detail


def read(
    values: np.ndarray,
    times: np.ndarray,
    side: float,
    flat: float,
    steeper: float,
    min_side: float,
) -> Reading:
    """The shape of ``values`` at every sample."""
    before, after = side_changes(values, times, side, min_side)
    pairs = [shape_at(b, a, flat, steeper) for b, a in zip(before, after, strict=True)]
    return Reading(tuple(s for s, _ in pairs), tuple(d for _, d in pairs))
