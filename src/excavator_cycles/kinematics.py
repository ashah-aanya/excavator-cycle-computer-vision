"""Fitting the excavator's kinematic chain to its silhouette.

An excavator arm is three rigid links -- boom, stick, bucket -- so its pose is
fully described by four points: where the boom meets the body, the boom-stick
joint, the stick-bucket joint, and the bucket tip. That is the parameterisation
the pose-estimation literature uses, and everything the state machine needs
follows from it.

Why not simply take the mask pixel farthest from the machine's centre, which is
what this pipeline did first: that measures distance *through the air*. Whenever
the arm is raised the boom's apex is farther from the pivot than the bucket
hanging below it, so the "bucket" lands on top of the boom -- through most of
the hauling phase, which is exactly when the measurement matters.

The fix is to measure **along the arm**. Geodesic distance from the boom base,
constrained to the mask, makes the bucket the end of the chain in every pose,
because a path to it must travel the whole arm however the arm is folded.

Published methods reach the same four points by training a keypoint network on
synthetic images from a game engine. That needs a rendering pipeline and a
training run; this needs the masks we already have. The trade is that a learned
model would be robust to a broken silhouette, while this is only as good as the
mask -- which is why the fit is checked against a physical constraint (§
``link_lengths``) rather than trusted.
"""

from __future__ import annotations

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)



def body_core(masks: list[np.ndarray], quantile: float = 0.9) -> np.ndarray:
    """The part of the machine that is always there: body, cab, undercarriage.

    The arm sweeps through the frame while the body stays put, so occupancy over
    time separates them with no detection of either.
    """
    occupancy = np.mean(np.stack(masks).astype(np.float32), axis=0)
    occupied = occupancy[occupancy > 0]
    if occupied.size == 0:
        return np.zeros_like(occupancy, dtype=bool)
    threshold = float(np.quantile(occupied, quantile))
    return occupancy >= max(threshold, 1e-6)


def boom_base(core: np.ndarray) -> tuple[float, float]:
    """Where the boom is anchored: the top of the persistent body.

    The boom pivots from the cab, which sits at the top of the body core. Taking
    the topmost band rather than the centroid matters because the centroid sits
    down among the tracks, and a base below the machine makes every arm path
    start by travelling up through the body.
    """
    ys, xs = np.nonzero(core)
    if ys.size == 0:
        raise ValueError("the body core is empty; cannot locate the boom base")
    cutoff = np.percentile(ys, 20)  # the upper fifth of the body
    upper = ys <= cutoff
    return float(xs[upper].mean()), float(ys[upper].mean())


def geodesic_distance(
    mask: np.ndarray, seed: tuple[float, float], max_steps: int = 2000
) -> np.ndarray:
    """Steps from ``seed`` to every mask pixel, travelling only within the mask.

    A wavefront: repeatedly dilate the reached region, clipped to the mask, and
    record the step at which each pixel is first touched. Distance measured this
    way follows the arm instead of cutting across empty space, which is the whole
    point -- a folded arm is far along the chain even when it is near in space.

    Unreached pixels are -1.
    """
    height, width = mask.shape
    distance = np.full((height, width), -1, dtype=np.int32)

    x, y = round(seed[0]), round(seed[1])
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))

    frontier = np.zeros((height, width), dtype=np.uint8)
    if mask[y, x]:
        frontier[y, x] = 1
    else:
        # The seed can fall just outside the mask when the body is thin; start
        # from the nearest mask pixel instead of failing.
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return distance
        index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        frontier[ys[index], xs[index]] = 1

    reached = frontier.copy()
    distance[frontier.astype(bool)] = 0
    kernel = np.ones((3, 3), np.uint8)

    for step in range(1, max_steps):
        grown = cv2.dilate(frontier, kernel) & mask.astype(np.uint8)
        frontier = grown & ~reached
        if not frontier.any():
            break
        distance[frontier.astype(bool)] = step
        reached |= frontier
    return distance








