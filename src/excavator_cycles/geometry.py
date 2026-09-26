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

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)



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


