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

    # Every split inside an empty gap scores IDENTICALLY: `weight_low`,
    # `weight_high` and both means stop changing the moment no mass lies between
    # the candidate and the next one. So the objective does not have a single
    # peak, it has a plateau spanning the gap -- and `argmax` resolves a tie at
    # the lowest index, which pins the threshold to the BOTTOM of the gap,
    # hugging the lower population. That error grows with the separation, so the
    # method fails hardest on exactly the cleanly-bimodal data it is best suited
    # to. The midpoint of the plateau is the conventional resolution and the only
    # one that is symmetric in the two groups.
    #
    # Exact equality, and it really is exact. Inside an empty gap the bin counts are
    # zero, so neither `weight_low` nor `cumulative` changes from one candidate to
    # the next and the scores are BIT-identical -- there is no rounding to absorb.
    #
    # An earlier version carried a machine-epsilon tolerance here, justified as
    # merging bins that "hold a few stray samples" and so differ only in the last
    # bits. That justification was wrong twice over: a relative tolerance of ~1e-14
    # cannot merge scores that differ because they contain different data, and
    # merging them would be incorrect anyway, since the midpoint rule below is only
    # valid across candidates that genuinely tie. Measured over 3000 random
    # distributions (Gaussian, bimodal, exponential, zero-inflated, Pareto) the
    # tolerance changed the answer in zero cases.
    peak = between.max()
    tied = np.flatnonzero(between == peak)
    return float((centres[tied[0]] + centres[tied[-1]]) / 2.0)
