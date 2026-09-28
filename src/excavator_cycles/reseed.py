"""Finding the bucket again after SAM 2 loses it.

Why this module exists
----------------------
The bucket is prompted **once** (``seeding.py``) and SAM 2 follows it from there.
That is right while the bucket stays in view, and wrong the first time it does
not. The bucket disappears as a matter of routine: it is lowered into the truck
bed and buried by what it is tipping, and it goes into the pile to dig. SAM 2
decides "is this my object?" by comparing each frame with a memory of recent
ones, so after about a second of nothing its memory is mostly nothing, and when
the bucket comes back out it is not claimed again. On ``random.mp4`` the bucket
went into the bed at 34.7 s and every sample to the end of the clip, 10.6 s of
them, had no bucket -- the confidence sat at ~0.08 while the bucket was plainly
visible.

What it does
------------
1. **Audit.** A sample is *missing* when it has no bucket mask or its confidence
   is below ``features.min_sample_confidence`` -- the floor the features stage
   already uses to treat a sample as a gap rather than a measurement, so a lost
   bucket means the same thing in both stages and there is no second threshold
   to keep in step. A run of missing samples lasting ``track.bucket_lost_seconds``
   or more is a *lost span*. Shorter runs are left alone: a blurred frame or a
   moment of dust is not worth a SAM pass.
2. **Reseed.** Inside each lost span, the same geometric ranking that chose the
   first seed picks the best frame, with one extra requirement: the bucket band
   must be clear of the truck, because over the bed is where the bucket is
   hidden. The frame where the loss was *noticed* is never used as such -- the
   ranking chooses, and it scores the frames where the arm is out and clear.
3. **Re-track, only that span.** A new SAM 2 session is prompted on that frame
   and propagated forward to the span's end and backward to its start. Samples
   outside the span are not touched.
4. **Once per span.** A span that overlaps one already attempted is not retried,
   so a bucket that stays buried is recorded as a gap rather than reseeded in a
   loop. ``track.bucket_max_reseeds`` bounds the total.

What it does not do
-------------------
It does not flag a *partly* hidden bucket. Mask area was the obvious signal, and
it was measured before being used: on runs where the bucket was tracked
correctly the area routinely falls below half its own median -- 1.6 s of vid2's
correctly tracked full cycle does, because the visible area changes with the
bucket's angle and its distance from the camera. A size rule would flag real
tipping, which is what the dump onset reads. See
``docs/stages/10-bucket-reseed.md``.

No model runs in this module. ``track()`` supplies the one function that does
(``track_span``), so the orchestration here is tested without weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_setup import get_logger
from .seeding import BucketSeed

log = get_logger(__name__)


@dataclass(frozen=True)
class LostSpan:
    """A stretch of samples with no usable bucket. Both ends inclusive, as ordinals."""

    start: int
    end: int

    def overlaps(self, other: LostSpan) -> bool:
        return self.start <= other.end and other.start <= self.end


def missing_samples(
    count: int,
    bucket_masks: dict[int, np.ndarray],
    bucket_confidences: dict[int, float],
    floor: float,
) -> np.ndarray:
    """True where a sample has no bucket mask, or one SAM 2 itself doubts.

    A NaN confidence means the model did not report one; that is unknown, not
    doubt, so it counts as present when a mask exists.
    """
    missing = np.ones(count, dtype=bool)
    for position, mask in bucket_masks.items():
        if 0 <= position < count and mask.any():
            confidence = bucket_confidences.get(position, float("nan"))
            missing[position] = bool(confidence < floor)  # NaN < floor is False
    return missing


def lost_spans(missing: np.ndarray, times: np.ndarray, min_seconds: float) -> list[LostSpan]:
    """Runs of missing samples that last at least ``min_seconds``.

    Duration is measured on the real timestamps plus one sample spacing, so a
    run of ten samples at 10 Hz lasts 1.0 s rather than 0.9 s.
    """
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    spans: list[LostSpan] = []
    position = 0
    while position < len(missing):
        if not missing[position]:
            position += 1
            continue
        end = position
        while end + 1 < len(missing) and missing[end + 1]:
            end += 1
        if times[end] - times[position] + step >= min_seconds:
            spans.append(LostSpan(position, end))
        position = end + 1
    return spans


def reseed_lost_spans(
    count: int,
    times: np.ndarray,
    bucket_masks: dict[int, np.ndarray],
    bucket_confidences: dict[int, float],
    floor: float,
    min_seconds: float,
    max_reseeds: int,
    find_seed: Callable[[range], BucketSeed | None],
    track_span: Callable[
        [BucketSeed, LostSpan], tuple[dict[int, np.ndarray], dict[int, float]]
    ],
) -> list[dict[str, Any]]:
    """Reseed every lost span once, updating the two bucket dicts in place.

    ``find_seed(positions)`` ranks frames inside a span; ``track_span(seed, span)``
    runs SAM 2 over that span alone. Returns one record per span attempted, for
    ``track.json``, including the spans where no frame qualified.
    """
    attempted: list[LostSpan] = []
    records: list[dict[str, Any]] = []
    while len(attempted) < max_reseeds:
        missing = missing_samples(count, bucket_masks, bucket_confidences, floor)
        spans = [
            s
            for s in lost_spans(missing, times, min_seconds)
            if not any(s.overlaps(a) for a in attempted)
        ]
        if not spans:
            break
        span = spans[0]
        attempted.append(span)
        record: dict[str, Any] = {
            "start_seconds": float(times[span.start]),
            "end_seconds": float(times[span.end]),
            "missing_before": int(missing[span.start : span.end + 1].sum()),
        }
        seed = find_seed(range(span.start, span.end + 1))
        if seed is None:
            log.info(
                "bucket lost %.1f-%.1f s; no frame in it shows the bucket clear of the "
                "truck, so it stays a gap",
                record["start_seconds"],
                record["end_seconds"],
            )
            record["seed_sample"] = None
            records.append(record)
            continue

        log.info(
            "bucket lost %.1f-%.1f s; reseeding at %.1f s",
            record["start_seconds"],
            record["end_seconds"],
            float(times[seed.sample]),
        )
        masks, confidences = track_span(seed, span)
        for position in range(span.start, span.end + 1):
            bucket_masks.pop(position, None)
            if position in masks:
                bucket_masks[position] = masks[position]
            if position in confidences:
                bucket_confidences[position] = confidences[position]

        after = missing_samples(count, bucket_masks, bucket_confidences, floor)
        record["seed_sample"] = seed.sample
        record["seed_seconds"] = float(times[seed.sample])
        record["missing_after"] = int(after[span.start : span.end + 1].sum())
        records.append(record)
    return records
