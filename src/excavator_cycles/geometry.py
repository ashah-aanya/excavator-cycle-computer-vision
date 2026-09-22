"""Stage 3: derive the machine's structure and the scene's landmarks from masks.

Only one thing was ever detected -- the excavator. Everything the state machine
needs is computed from the shape of that mask over time:

    rotation centre   the pixels that are machine in almost every frame
    scale L           how far the arm reaches, in this video's pixels
    bucket tip        the mask point farthest from the centre
    dig / dump zones  where the bucket lingers, clustered into two
    surface height    the split in the bimodal distribution of bucket elevation

That indirection is deliberate. Detectors are unreliable on object *parts* (an
arm) and on amorphous terrain (a soil pile); they are reliable on a whole
machine. So detect the easy thing and compute the rest.

Everything here is scale-free. Distances are divided by ``L``, so the same
numbers come out whether the camera is near or far, and no value measured in
pixels escapes this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .config import Config
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Scene:
    """Landmarks derived once per video, shared by every sample."""

    centre: tuple[float, float]  # rotation centre, pixels
    scale: float  # L: arm reach, pixels
    dig_zone: tuple[float, float] | None  # bucket dwell cluster, lower
    dump_zone: tuple[float, float] | None  # bucket dwell cluster, higher
    surface_height: float | None  # in units of L, relative to the centre
    return_sign: float  # +1 or -1: which way is "back to the pile"
    zone_separation: float | None  # distance between zones, in units of L

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rotation_centre(masks: list[np.ndarray], quantile: float) -> tuple[float, float]:
    """Where the machine pivots.

    An excavator's body and undercarriage stay put while the arm sweeps, so the
    pixels that are machine in *almost every* frame are the body. Their centroid
    is the pivot -- no detection of the body required, just arithmetic over the
    masks already computed.
    """
    occupancy = np.mean(np.stack(masks).astype(np.float32), axis=0)
    threshold = (
        float(np.quantile(occupancy[occupancy > 0], quantile)) if occupancy.any() else 0.0
    )
    core = occupancy >= max(threshold, 1e-6)
    if not core.any():
        core = occupancy > 0
    ys, xs = np.nonzero(core)
    return float(xs.mean()), float(ys.mean())


def farthest_point(
    mask: np.ndarray, centre: tuple[float, float]
) -> tuple[float, float] | None:
    """The bucket end: the mask pixel farthest from the pivot.

    An excavator is a body with one long thing sticking out of it, so the far end
    of the shape is the bucket. This is a straight-line distance rather than a
    distance measured along the mask; the two agree while the arm is extended and
    can disagree when it folds back over the body, which is why the caller gates
    sudden jumps rather than believing every frame.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    distances = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2
    index = int(np.argmax(distances))
    return float(xs[index]), float(ys[index])


def arm_reach(
    tips: list[tuple[float, float] | None], centre: tuple[float, float], percentile: float
) -> float:
    """``L``: the scale unit, in this video's pixels.

    A high percentile rather than the maximum, so a single bad frame cannot set
    the scale for the whole video. Every distance downstream is divided by this,
    which is the entire reason a hidden video shot from a different distance
    needs no special handling.
    """
    distances = [
        float(np.hypot(tip[0] - centre[0], tip[1] - centre[1]))
        for tip in tips
        if tip is not None
    ]
    if not distances:
        raise ValueError("no valid bucket positions; cannot establish a scale")
    return float(np.percentile(distances, percentile))


def bucket_region(mask: np.ndarray, tip: tuple[float, float], radius: float) -> np.ndarray:
    """The part of the machine near the far end -- the bucket and a little stick."""
    ys, xs = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    near = (xs - tip[0]) ** 2 + (ys - tip[1]) ** 2 <= radius**2
    return mask & near


def bucket_axis(region: np.ndarray) -> tuple[float, float] | None:
    """Orientation of the bucket region's long axis, and how elongated it is.

    Returns ``(angle, elongation)`` where the angle is in radians **modulo pi**
    -- an axis has no head or tail, so 10 degrees and 190 degrees are the same
    axis -- and elongation is the ratio of the long side to the short one.

    That ratio is not decoration. When a shape is nearly as wide as it is long,
    its "long axis" is nearly arbitrary: a couple of pixels flipping swings the
    answer by 90 degrees. At 480x272 the bucket is often in that regime, so the
    caller must be able to reject the measurement instead of receiving a
    confident wrong number.
    """
    points = cv2.findNonZero(region.astype(np.uint8))
    if points is None or len(points) < 5:
        return None
    (_, _), (width, height), angle = cv2.minAreaRect(points)
    if width <= 0 or height <= 0:
        return None
    long_side, short_side = max(width, height), min(width, height)
    if width < height:  # report the LONG axis, whichever side that is
        angle += 90.0
    return float(np.deg2rad(angle) % np.pi), float(long_side / short_side)


def relative_angle(axis: float, reference: float) -> float:
    """Angle between two axes, wrapped to (-pi/2, pi/2].

    The bucket's curl is meaningful only *relative to the forearm*: the absolute
    axis swings through a whole revolution as the arm slews, which buries the
    small rotation that actually signals dumping. Both quantities are axes
    rather than directions, so the difference wraps at pi, not 2pi.
    """
    delta = (axis - reference) % np.pi
    return float(delta - np.pi if delta > np.pi / 2 else delta)


def unwrap_axis(values: np.ndarray) -> np.ndarray:
    """Unwrap a sequence of axis angles, which repeat every pi.

    ``np.unwrap`` assumes a period of 2*pi. Applied to an axis it treats the
    meaningless 180-degree flips as real motion, which is what made the curl
    signal look like noise in the first plots.
    """
    return np.unwrap(values, period=np.pi)


def dwell_clusters(
    tips: list[tuple[float, float] | None], speeds: np.ndarray, config: Config, scale: float
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, float | None]:
    """Where the bucket lingers: the dig location and the dump location.

    Loading work is strongly bimodal -- the bucket dwells where it digs and where
    it dumps, and moves briskly in between -- so clustering the slow samples
    recovers both without detecting soil or a truck. The lower cluster is the dig
    zone, because material is picked up below and released above.

    Returns ``(None, None, None)`` when the two clusters are not convincingly
    apart, which is a legitimate outcome to report rather than paper over.
    """
    from scipy.cluster.vq import kmeans2

    valid = np.array(
        [
            (position[0], position[1], speed)
            for position, speed in zip(tips, speeds, strict=True)
            if position is not None and np.isfinite(speed)
        ]
    )
    if len(valid) < 10:
        log.warning("too few valid samples to locate the dig and dump zones")
        return None, None, None

    cutoff = np.quantile(valid[:, 2], config.geometry.dwell_speed_quantile)
    dwell = valid[valid[:, 2] <= cutoff][:, :2]
    if len(dwell) < 4:
        log.warning("too few dwell samples to locate the dig and dump zones")
        return None, None, None

    centroids, labels = kmeans2(dwell, 2, minit="++", seed=0)
    if len(np.unique(labels)) < 2:
        log.warning("bucket dwell positions did not separate into two clusters")
        return None, None, None

    # Image y grows downward, so the larger y is physically lower: the dig side.
    lower, upper = sorted(centroids, key=lambda c: -c[1])
    separation = float(np.hypot(*(np.array(lower) - np.array(upper))) / scale)
    if separation < config.geometry.min_zone_separation:
        log.warning(
            "dig and dump zones are only %.2f L apart; treating as unresolved", separation
        )
        return None, None, separation

    return (float(lower[0]), float(lower[1])), (float(upper[0]), float(upper[1])), separation


def surface_height(elevations: np.ndarray, in_dig_zone: np.ndarray) -> float | None:
    """The material surface, found where the bucket digs.

    Inside the dig zone the bucket is either below the surface (digging) or above
    it (approaching and lifting), so its elevation is two-humped and the split
    between the humps *is* the surface. Otsu's method finds that split with no
    threshold chosen by anyone.
    """
    values = elevations[in_dig_zone & np.isfinite(elevations)]
    if values.size < 10:
        log.warning("too few samples in the dig zone to estimate the surface height")
        return None
    return float(otsu_threshold(values))


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """The split that best separates a two-humped distribution.

    Otsu's method: try every candidate split and keep the one maximising the
    variance *between* the two groups. Parameter-free, which is the point -- a
    hand-chosen surface height would be exactly the kind of video-specific
    constant the task forbids.
    """
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    total = counts.sum()
    if total == 0:
        return float(np.median(values))

    weight_low = np.cumsum(counts)
    weight_high = total - weight_low
    valid = (weight_low > 0) & (weight_high > 0)
    if not valid.any():
        return float(np.median(values))

    cumulative = np.cumsum(counts * centres)
    mean_low = np.divide(
        cumulative, weight_low, out=np.zeros_like(cumulative), where=weight_low > 0
    )
    mean_high = np.divide(
        cumulative[-1] - cumulative,
        weight_high,
        out=np.zeros_like(cumulative),
        where=weight_high > 0,
    )
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -np.inf
    return float(centres[int(np.argmax(between))])


def return_direction(dig_zone, dump_zone, centre: tuple[float, float]) -> float:
    """Which rotational direction means "back toward the pile".

    Derived per video, so a clip with the truck on the opposite side works with
    no change -- the pipeline is mirror-invariant for free. Returns +1 or -1.
    """
    if dig_zone is None or dump_zone is None:
        return 1.0
    dig_bearing = np.arctan2(-(dig_zone[1] - centre[1]), dig_zone[0] - centre[0])
    dump_bearing = np.arctan2(-(dump_zone[1] - centre[1]), dump_zone[0] - centre[0])
    delta = np.arctan2(np.sin(dig_bearing - dump_bearing), np.cos(dig_bearing - dump_bearing))
    return float(np.sign(delta)) or 1.0
