"""Finding the bucket in one frame, so SAM 2 can be told where to look.

Why this module exists
----------------------
SAM 2 does not recognise anything. It segments whatever it is pointed at and
then tracks that. So the only question is the cheapest reliable way to point at
the bucket **once**.

Asking a detector was measured and does not work: nineteen prompt/detector
combinations across Grounding DINO and OWLv2, and the best-scoring one
("excavator bucket.", found on every frame at 0.645) boxes the **whole
machine** -- it matched the word *excavator*. See
``docs/stages/06-bucket-mask.md``.

Geometry works instead, because of what a bucket *is*: the far end of the arm.
Two properties make this safe here even though a nearly identical operation --
``farthest_point`` per frame -- was the first of four failed attempts at arm
pose:

* it happens **once**, on a frame this module gets to choose, rather than being
  re-derived on all 296 and then differentiated;
* the distance is **geodesic**, measured along the metal rather than through
  the air, so a raised boom cannot outrank the bucket. Euclidean distance puts
  the seed on the boom apex whenever the arm is up -- measured at 70 px away
  from the correct answer through the dumping window.

What is returned
----------------
Three descriptions of the same region, because SAM 2's own ablation (paper
Table 4, zero-shot VOS averaged over 17 datasets) prices them very differently:

    1 click   64.3 J&F      <- do not do this
    box       72.9
    3 clicks  73.2
    5 clicks  75.4
    mask      77.6          <- the band itself is a mask

So ``BucketSeed`` carries points, a box and the band, and the caller picks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .kinematics import arm_region, geodesic_distance

log = logging.getLogger(__name__)


@dataclass
class BucketSeed:
    """Where the bucket is on one chosen frame, in every form SAM 2 accepts."""

    sample: int
    """Sample ordinal of the frame this was derived from."""

    points: np.ndarray
    """(n, 2) positive points, interior to the bucket, as (x, y)."""

    negatives: np.ndarray
    """(m, 2) points on the rest of the machine, to stop SAM claiming the arm."""

    box: tuple[int, int, int, int]
    """(x0, y0, x1, y1) around the band."""

    band: np.ndarray
    """The geodesic band itself -- the strongest prompt SAM 2 takes."""

    score: float
    """How good this frame looked, for comparing candidates. Higher is better."""

    reach: float
    """Geodesic extent of the arm on this frame, in steps."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BucketSeed(sample={self.sample}, points={len(self.points)}, "
            f"box={self.box}, score={self.score:.3f})"
        )


def geodesic_bands(
    mask: np.ndarray,
    core: np.ndarray,
    pivot: tuple[float, float],
    bucket_from: float = 0.82,
    stick_from: float = 0.50,
    stick_to: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split the arm into a bucket band and a stick band by distance along it.

    The body is removed first. That is the whole reason this works where a
    Euclidean radius band did not: the cab and counterweight sit at *small*
    radius, so a radius band cuts straight through them and the inner band ends
    up mostly cab -- which does not rotate about the pivot in projection, so it
    read ~0 rotation in every phase.

    Returns ``(bucket, stick, reach)``; both bands are empty and ``reach`` is
    0.0 when the arm cannot be traced.
    """
    empty = np.zeros_like(mask, dtype=bool)
    arm = arm_region(mask, core)
    if not arm.any():
        return empty, empty, 0.0

    distance = geodesic_distance(arm, pivot)
    reachable = distance >= 0
    if not reachable.any():
        return empty, empty, 0.0

    reach = float(distance[reachable].max())
    if reach <= 0:
        return empty, empty, 0.0

    bucket = reachable & (distance >= bucket_from * reach)
    stick = reachable & (distance >= stick_from * reach) & (distance < stick_to * reach)
    return bucket, stick, reach


def interior_points(region: np.ndarray, count: int, min_pixels: int = 25) -> np.ndarray:
    """Pick well-spread points that sit *inside* a region, never on its edge.

    Mask boundaries are the least trustworthy part of a mask -- they deform by
    10-30 px whenever parts overlap. So points are chosen by distance transform:
    the first is the deepest interior pixel, and each next one is the deepest
    pixel that is also far from those already chosen. That is farthest-point
    sampling weighted by depth, and it degrades gracefully -- a slightly wrong
    region still yields points near its middle.

    Returns an ``(n, 2)`` array of (x, y); ``n`` may be fewer than ``count`` for
    a small region, and the array is empty if the region is too small to trust.
    """
    if region.sum() < min_pixels or count < 1:
        return np.zeros((0, 2), dtype=np.float64)

    depth = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    chosen: list[tuple[float, float]] = []
    ys, xs = np.nonzero(region)
    candidates = np.column_stack([xs, ys]).astype(np.float64)
    weights = depth[ys, xs].astype(np.float64)

    for _ in range(count):
        if not len(candidates):
            break
        if chosen:
            taken = np.asarray(chosen)
            spread = np.min(
                np.linalg.norm(candidates[:, None, :] - taken[None, :, :], axis=2), axis=1
            )
            # Depth keeps points off the edge; spread keeps them from clumping.
            merit = weights * (1.0 + spread)
        else:
            merit = weights
        index = int(np.argmax(merit))
        chosen.append((float(candidates[index, 0]), float(candidates[index, 1])))
        weights[index] = -1.0

    return np.asarray(chosen, dtype=np.float64)


def _bounding_box(region: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(region)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _compactness(region: np.ndarray) -> float:
    """Area over bounding-box area. A bucket is a blob; a smear of arm is not."""
    x0, y0, x1, y1 = _bounding_box(region)
    box_area = max((x1 - x0) * (y1 - y0), 1)
    return float(region.sum()) / box_area


def score_frame(
    bucket: np.ndarray,
    reach: float,
    scale: float,
    truck_box: tuple[float, float, float, float] | None,
    position: int,
    total: int,
) -> float:
    """How good a frame is to seed from. Higher is better; 0 means unusable.

    Four things matter, and each was learned the hard way:

    * **reach** -- an extended arm makes the far end unambiguous;
    * **compactness** -- a band that is a blob is the bucket, a band that is a
      long smear is a slice of stick;
    * **clearance from the truck** -- seeding next to the truck invites SAM to
      claim truck pixels, and the truck box is free and static;
    * **mid-clip position** -- SAM propagates both ways from the seed, so a
      frame at the very end makes the whole video one long reverse pass. The
      naive "maximum reach" rule picked the *last* frame of the clip.
    """
    if not bucket.any() or reach <= 0:
        return 0.0

    _, xs = np.nonzero(bucket)
    centre_x = float(xs.mean())

    clearance = 1.0
    if truck_box is not None:
        gap = min(abs(centre_x - truck_box[0]), abs(centre_x - truck_box[2]))
        inside = truck_box[0] <= centre_x <= truck_box[2]
        clearance = 0.0 if inside else min(gap / max(scale * 0.5, 1.0), 1.0)

    midness = 1.0 - abs(position - total / 2) / max(total / 2, 1)
    return float(reach / max(scale, 1.0) + _compactness(bucket) + clearance + 0.5 * midness)


def choose_seed(
    masks: list[np.ndarray],
    core: np.ndarray,
    pivot: tuple[float, float],
    scale: float,
    truck_box: tuple[float, float, float, float] | None = None,
    point_count: int = 5,
    negative_count: int = 3,
    bucket_from: float = 0.82,
    stick_from: float = 0.50,
    stick_to: float = 0.75,
) -> BucketSeed | None:
    """Score every frame and return the best place to point SAM 2 at the bucket.

    Returns ``None`` when no frame yields a usable band -- the caller decides
    whether that is fatal. Candidates are ranked rather than thresholded, so
    there is always a best frame if there is any frame at all.
    """
    best: BucketSeed | None = None
    usable = 0

    for position, mask in enumerate(masks):
        if mask is None or not mask.any():
            continue
        bucket, stick, reach = geodesic_bands(
            mask, core, pivot, bucket_from, stick_from, stick_to
        )
        points = interior_points(bucket, point_count)
        if not len(points):
            continue
        usable += 1

        score = score_frame(bucket, reach, scale, truck_box, position, len(masks))
        if best is not None and score <= best.score:
            continue

        negatives = interior_points(stick, negative_count)
        best = BucketSeed(
            sample=position,
            points=points,
            negatives=negatives,
            box=_bounding_box(bucket),
            band=bucket,
            score=score,
            reach=reach,
        )

    if best is None:
        log.warning("no frame produced a usable bucket band from %d masks", len(masks))
        return None

    log.info(
        "bucket seed: sample %d of %d (%d candidates), score %.3f, %d points, box %s",
        best.sample,
        len(masks),
        usable,
        best.score,
        len(best.points),
        best.box,
    )
    return best
