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

from .kinematics import geodesic_distance

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

    The body is excluded from the bands. That is the whole reason this works
    where a Euclidean radius band did not: the cab and counterweight sit at
    *small* radius, so a radius band cuts straight through them and the inner
    band ends up mostly cab -- which does not rotate about the pivot in
    projection, so it read ~0 rotation in every phase.

    The wavefront runs over the **whole** mask, not over the arm alone, and the
    body is subtracted only when the bands are formed. That ordering matters:
    subtracting the body first shatters the mask into four to six pieces, and
    since the wavefront starts at the pivot -- which is *inside* the removed
    body -- it begins in whichever fragment happens to be nearest and is trapped
    there. Measured on the development video, that left ``reach = 1`` on frames
    carrying 5,000 arm pixels, and the band collapsed to a single pixel. Running
    over the connected mask and masking afterwards halved the rate at which the
    band teleports, from 7% of samples to 4%.

    Travelling through the body costs almost nothing, because it is a compact
    blob sitting on the pivot, so distance along the arm still grows
    monotonically outward.

    Returns ``(bucket, stick, reach)``; both bands are empty and ``reach`` is
    0.0 when the arm cannot be traced.
    """
    empty = np.zeros_like(mask, dtype=bool)
    if not mask.any():
        return empty, empty, 0.0

    distance = geodesic_distance(mask, pivot)
    reachable = distance >= 0
    if not reachable.any():
        return empty, empty, 0.0

    reach = float(distance[reachable].max())
    if reach <= 0:
        return empty, empty, 0.0

    body = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    live = reachable & ~body

    bucket = live & (distance >= bucket_from * reach)
    stick = live & (distance >= stick_from * reach) & (distance < stick_to * reach)
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


def looks_like_the_boom_apex(
    bucket: np.ndarray, mask: np.ndarray, scale: float, margin_fraction: float = 0.08
) -> bool:
    """Is this band on the top of the machine rather than on the bucket?

    The one failure geodesic distance genuinely has. When the arm folds far
    enough that the bucket touches the cab *in projection*, the wavefront takes
    the shortcut and the farthest point along the shape becomes the boom apex.
    The shape really is ambiguous there -- two parts of the machine are in
    contact -- so no band edge fixes it.

    On an excavator the apex is the top of the silhouette by construction, and
    the bucket hangs at the end of the arm below it. Measured on the development
    video this fires on 19% of hauling and, importantly, on **0 of 44** dumping
    samples -- so it does not punish a legitimately raised bucket, because the
    boom is still above it.

    Reach alone does not catch this. The short-circuit shortens the path, so bad
    frames do score lower on average (median reach 108 against 158), but the
    worst of them still reach 223 -- as high as a good frame.

    The margin is a fraction of ``scale`` (the arm's reach), never a pixel
    count. A video shot from twice as far would make any pixel constant mean
    something different, and a constant fitted to one clip is the thing this
    project is not allowed to ship. 0.08 L is ~18 px on the development video.
    """
    if not bucket.any() or not mask.any():
        return False
    gap = float(np.nonzero(bucket)[0].mean() - np.nonzero(mask)[0].min())
    return gap < margin_fraction * max(scale, 1.0)


def score_frame(
    bucket: np.ndarray,
    reach: float,
    scale: float,
    truck_box: tuple[float, float, float, float] | None,
    position: int,
    total: int,
    mask: np.ndarray | None = None,
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

    Passing ``mask`` additionally rejects frames where the band has landed on
    the boom apex, which is the one failure geodesic distance genuinely has;
    see ``looks_like_the_boom_apex``. Without it, the best such frame still
    ranked #110 of 287 -- demoted, but not excluded.
    """
    if not bucket.any() or reach <= 0:
        return 0.0
    if mask is not None and looks_like_the_boom_apex(bucket, mask, scale):
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


FRAME_RULES = ("score", "reach")


def extension(mask: np.ndarray, pivot: tuple[float, float]) -> float:
    """Straight-line distance from the pivot to the farthest pixel of the mask."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    return float(np.hypot(xs - pivot[0], ys - pivot[1]).max())


def reach_score(
    bucket: np.ndarray, mask: np.ndarray, pivot: tuple[float, float], scale: float
) -> float:
    """Frame rule "reach": the most fully stretched arm wins. 0 means unusable.

    The bucket hangs at the end of the arm, so the band (the far end of the walk along the
    metal) is the bucket when the arm is stretched out, and it is a slice of stick when the
    arm is folded or the bucket is missing from the mask. How far the mask reaches from the
    pivot in a straight line separates those cases. Measured offline on six runs, ranking by
    this alone picked a frame whose band overlaps the real bucket at 0.83 (five sound runs),
    against the worst score of everything tried for the score in ``score_frame``.

    This is used to RANK frames only. It is the trap ``farthest_point`` fell into when used
    to place the band, so the band itself is still the geodesic one, and a frame whose band
    is on the boom apex still scores 0, which is what keeps a raised boom from winning.
    Nothing here is a pixel constant: ``scale`` only turns the value into a fraction of the
    arm's reach so it is comparable between videos, and it does not change the ranking.
    """
    if not bucket.any() or not mask.any() or scale <= 0:
        return 0.0
    if looks_like_the_boom_apex(bucket, mask, scale):
        return 0.0
    return extension(mask, pivot) / max(scale, 1.0)


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
    positions: range | None = None,
    clear_of_truck: bool = False,
    rule: str = "score",
) -> BucketSeed | None:
    """Score every frame and return the best place to point SAM 2 at the bucket.

    ``rule`` says how frames are ranked: "score" is ``score_frame`` (reach along the metal,
    compactness, clearance from the truck, mid-clip-ness); "reach" is ``reach_score`` (how
    far the arm is stretched, straight-line from the pivot).

    Returns ``None`` when no frame yields a usable band -- the caller decides
    whether that is fatal. Candidates are ranked rather than thresholded, so
    there is always a best frame if there is any frame at all.

    ``positions`` limits the search to one stretch of the clip, and "mid-clip"
    then means the middle of that stretch: a reseed propagates both ways inside
    the stretch, exactly as the first seed does inside the clip. ``clear_of_truck``
    turns the truck clearance from a penalty into a requirement. A reseed is
    needed because the bucket vanished, and over the bed is where it vanishes --
    lowered in, buried by what it is tipping -- so a band there is the arm
    reaching into the bed, not the bucket.
    """
    if rule not in FRAME_RULES:
        raise ValueError(f"bucket frame rule is {rule!r}; expected one of {FRAME_RULES}")
    best: BucketSeed | None = None
    usable = 0
    span = positions if positions is not None else range(len(masks))

    for position in span:
        mask = masks[position]
        if mask is None or not mask.any():
            continue
        bucket, stick, reach = geodesic_bands(
            mask, core, pivot, bucket_from, stick_from, stick_to
        )
        points = interior_points(bucket, point_count)
        if not len(points):
            continue
        if clear_of_truck and truck_box is not None:
            # Every pixel, not the centre: a band split between the bucket and
            # something over the truck has its centre between the two, clear of
            # the truck, and would be handed to SAM 2 whole.
            x0, y0, x1, y1 = (round(v) for v in truck_box)
            if bucket[max(y0, 0) : max(y1 + 1, 0), max(x0, 0) : max(x1 + 1, 0)].any():
                continue
        usable += 1

        if rule == "reach":
            score = reach_score(bucket, mask, pivot, scale)
        else:
            score = score_frame(
                bucket, reach, scale, truck_box, position - span.start, len(span), mask
            )
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
        log.warning(
            "no frame produced a usable bucket band from %d candidate frames%s",
            len(span),
            " clear of the truck" if clear_of_truck and truck_box is not None else "",
        )
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
