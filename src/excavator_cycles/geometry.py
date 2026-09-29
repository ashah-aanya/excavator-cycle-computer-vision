"""Stage 3: derive the machine's structure and the scene's landmarks from masks.

Only one thing was ever detected -- the excavator. Everything the phase search
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


def occupancy(masks: list[np.ndarray]) -> np.ndarray:
    """Per-pixel fraction of the frames in which that pixel was machine.

    A long exposure. Photograph a ceiling fan for a second and the hub stays
    sharp while the blades smear, because brightness records the fraction of
    time something was there. The same arithmetic separates an excavator's body
    from its arm: the body holds still while the arm sweeps, so persistence
    tells them apart with no detection of either.

    Float in [0, 1], same shape as one mask.
    """
    if not masks:
        raise ValueError("no masks given; cannot compute occupancy")
    return np.mean(np.stack(masks).astype(np.float32), axis=0)


def stable_core(masks: list[np.ndarray], quantile: float) -> tuple[np.ndarray, float]:
    """The pixels that are machine in almost every frame -- the body.

    Args:
        quantile: a position in *this video's* occupancy distribution, not a
            pixel value and not an occupancy level. 0.90 means "keep the most
            persistent tenth of the pixels that were ever machine", and what
            cutoff that works out to differs between videos by design.

    Returns the boolean core and the cutoff, so a caller can report what the
    quantile actually meant here.
    """
    occ = occupancy(masks)
    ever = occ[occ > 0]
    if ever.size == 0:
        raise ValueError("masks are empty; no machine pixels anywhere")
    threshold = float(np.quantile(ever, quantile))
    core = occ >= max(threshold, float(np.finfo(np.float32).tiny))
    if not core.any():
        # Only reachable when every pixel shares one occupancy value. Fall back
        # to "was ever machine" rather than returning nothing.
        log.warning("occupancy quantile %.2f produced an empty core; using all", quantile)
        core = occ > 0
    return core, threshold


def rotation_centre(masks: list[np.ndarray], quantile: float) -> tuple[float, float]:
    """Where the machine pivots: the centroid of the pixels that never move.

    A centroid rather than the centre of a box around them, because a box is
    defined by its most marginal pixel. Measured across a quantile sweep wide
    enough to change the core's size 4.4x, the centroid moves 10.5 px in x while
    a box's width moves 163 px.
    """
    core, _threshold = stable_core(masks, quantile)
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
