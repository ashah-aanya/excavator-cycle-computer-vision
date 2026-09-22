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

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Chain:
    """The arm's pose in one frame, as the four canonical points.

    ``None`` everywhere means the fit failed for this frame, which is a state the
    caller must handle: a missing pose is honest, a guessed one is not.
    """

    base: tuple[float, float]
    boom_stick: tuple[float, float]
    stick_bucket: tuple[float, float]
    tip: tuple[float, float]
    path_length: float  # along the arm, in pixels
    residual: float  # how far the path strays from the fitted segments

    @property
    def points(self) -> list[tuple[float, float]]:
        return [self.base, self.boom_stick, self.stick_bucket, self.tip]

    def link_lengths(self) -> tuple[float, float, float]:
        """Boom, stick and bucket lengths implied by this frame's fit.

        These are the physical constraint: the boom does not change length, so a
        frame whose fit disagrees with the video's median is a bad fit, not a
        machine that grew.
        """
        points = self.points
        return tuple(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in pairwise(points))


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


def trace_path(distance: np.ndarray, end: tuple[int, int]) -> list[tuple[int, int]]:
    """Walk from a point back to the seed, downhill through the distance map.

    Each step moves to a neighbour one closer to the seed, so the result is a
    shortest path along the arm: base first, tip last.
    """
    y, x = end
    path = [(x, y)]
    height, width = distance.shape

    while distance[y, x] > 0:
        target = distance[y, x] - 1
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and distance[ny, nx] == target:
                    y, x = ny, nx
                    path.append((x, y))
                    found = True
                    break
            if found:
                break
        if not found:  # shouldn't happen, but never loop forever on odd masks
            break
    return path[::-1]


def fit_chain(path: list[tuple[int, int]], min_length: int = 12) -> Chain | None:
    """Fit three straight links to the arm's path.

    The path bends where the joints are, so simplifying it to four vertices
    recovers them. Douglas-Peucker does the simplification; if it returns more
    vertices than needed, the ones contributing least to the shape are dropped
    until four remain.
    """
    if len(path) < min_length:
        return None

    points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
    total = cv2.arcLength(points, closed=False)

    # Start tight and relax until the path reduces to at most four vertices.
    for tolerance in (0.01, 0.02, 0.04, 0.08, 0.15):
        simplified = cv2.approxPolyDP(points, tolerance * total, closed=False)
        vertices = simplified.reshape(-1, 2)
        if len(vertices) <= 4:
            break

    if len(vertices) < 2:
        return None
    while len(vertices) > 4:  # drop the least significant corner
        angles = [
            _turn_angle(vertices[i - 1], vertices[i], vertices[i + 1])
            for i in range(1, len(vertices) - 1)
        ]
        vertices = np.delete(vertices, int(np.argmin(angles)) + 1, axis=0)
    while len(vertices) < 4:  # split the longest link to keep four points
        lengths = [
            np.hypot(*(vertices[i + 1] - vertices[i])) for i in range(len(vertices) - 1)
        ]
        index = int(np.argmax(lengths))
        midpoint = ((vertices[index] + vertices[index + 1]) / 2).astype(np.int32)
        vertices = np.insert(vertices, index + 1, midpoint, axis=0)

    chain = Chain(
        base=tuple(map(float, vertices[0])),
        boom_stick=tuple(map(float, vertices[1])),
        stick_bucket=tuple(map(float, vertices[2])),
        tip=tuple(map(float, vertices[3])),
        path_length=float(total),
        residual=_residual(np.array(path, dtype=float), vertices.astype(float)),
    )
    return chain


def _turn_angle(a, b, c) -> float:
    """How sharply the path turns at b; a straight run turns by 0."""
    first, second = np.array(b) - np.array(a), np.array(c) - np.array(b)
    n1, n2 = np.linalg.norm(first), np.linalg.norm(second)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / (n1 * n2), -1, 1))
    return float(np.arccos(cosine))


def _residual(path: np.ndarray, vertices: np.ndarray) -> float:
    """Mean distance from the traced path to the fitted segments.

    A large residual means the three-link model does not describe this
    silhouette -- a broken mask, or an arm hidden behind the cab -- so the frame
    can be rejected instead of yielding a confident wrong pose.
    """
    distances = []
    for point in path:
        best = min(
            _point_segment_distance(point, vertices[i], vertices[i + 1])
            for i in range(len(vertices) - 1)
        )
        distances.append(best)
    return float(np.mean(distances)) if distances else float("inf")


def _point_segment_distance(point, start, end) -> float:
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0:
        return float(np.hypot(*(point - start)))
    t = float(np.clip((point - start) @ segment / length_squared, 0, 1))
    return float(np.hypot(*(point - (start + t * segment))))


def fit_frame(mask: np.ndarray, base: tuple[float, float]) -> Chain | None:
    """Fit the chain in one frame: geodesic distance, trace, then three links."""
    if not mask.any():
        return None
    distance = geodesic_distance(mask, base)
    if not (distance > 0).any():
        return None
    # The geodesically farthest point is the end of the arm in ANY pose -- the
    # property straight-line distance does not have.
    index = int(np.argmax(distance))
    end = np.unravel_index(index, distance.shape)
    return fit_chain(trace_path(distance, (int(end[0]), int(end[1]))))
