"""Measuring what the machine is doing from the motion of its pixels.

Why this module exists
----------------------
Everything in ``kinematics.py`` answers the question *where is the arm?* and
then differentiates the answer to find out *what is it doing?*. That route has
failed repeatedly, and the reason is structural rather than a bug to be found:

* locating a joint means picking a **few parameters from a few pixels** by a
  hard choice -- an argmax, a corner, a breakpoint -- and a hard choice flips
  discontinuously when the evidence is marginal;
* recovering motion by **differencing** positions amplifies whatever noise that
  flip introduced.

This module inverts both. It measures motion **directly** from dense optical
flow, and reduces thousands of flow vectors to one number with a **median**.
An average degrades by a few percent when some of its inputs are wrong; an
argmax relocates. Measured on the development video, the arm-pose route produced
slew rates of +/-2 to 3.5 rad/s -- 150 to 200 degrees per second, which no
excavator can slew at -- while the same moments measured here stay inside
+/-0.5 rad/s and reverse sign exactly where the machine reverses direction.

Evidence: ``docs/stages/04-motion-field-probe.md``.

What is computed, per pair of consecutive samples
-------------------------------------------------
Dense flow ``F`` over the frame, then restricted to the tracked mask. With ``r``
the offset of each pixel from the slew pivot:

=========== ======================================= ==========================
symbol      definition                               what it is evidence for
=========== ======================================= ==========================
``omega``   median of ``(r x F) / |r|^2``            swing rate; its SIGN
                                                     separates hauling from
                                                     returning
``v_rad``   median of ``(r . F) / |r|``              reaching out vs folding in
``v_up``    median of ``-F_y`` over the far end      bucket rising or falling
``speed``   median of ``|F|``                        moving at all, or dwelling
=========== ======================================= ==========================

Two deliberate choices, both physical rather than tuned:

**Aggregate over the moving part, not the whole silhouette.** The tracks and
the body occupy a large share of the mask and do not rotate about the pivot in
this projection, so a median over everything is diluted towards zero by pixels
that are stationary by construction. Measured on the development video during a
swing: 0.005 rad/s over the whole mask against 0.257 rad/s over the outer half
-- the same motion, fifty times weaker, because most of the mask was not
participating in it. ``radial_fraction`` sets where "the moving part" starts.

**Erode the mask before sampling flow.** Flow on the silhouette's edge mixes
machine pixels with background pixels and reports something that is neither.

Units follow the rest of the pipeline: ``omega`` in rad/s, the linear rates in
reaches (``L``) per second, so one set of thresholds serves any video at any
scale or frame rate.

Sign convention matches ``features.bearing``, which is ``atan2(-(y-cy), x-cx)``
-- image ``y`` points down, so it is negated to give the usual anticlockwise
positive sense. Verified against motion visible in the frames rather than
assumed: the loaded swing and the empty return come out with opposite signs, and
``v_up`` is positive while the bucket climbs out of the cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class MotionField:
    """Per-sample motion summaries. Every array has one entry per sample.

    The last entry of each is NaN: a flow field is measured *between* two
    samples, so ``n`` samples yield ``n - 1`` measurements.
    """

    omega: np.ndarray  # swing rate about the pivot, rad/s, anticlockwise positive
    v_radial: np.ndarray  # reaching out (+) or folding in (-), L/s
    v_up: np.ndarray  # far end of the arm rising (+) or falling (-), L/s
    speed: np.ndarray  # median flow magnitude in the mask, L/s
    coverage: np.ndarray  # fraction of mask pixels carrying measurable flow

    def __len__(self) -> int:
        return len(self.omega)


def dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Farneback optical flow between two frames, as ``(H, W, 2)`` in pixels.

    Farneback rather than a learned flow network for two reasons that matter
    here: it needs no weights to download or license, and it is dense, which is
    the whole point -- a sparse tracker returns a handful of points and puts us
    back in the business of trusting individual locations.

    The frames are converted to grayscale if they are not already; flow is a
    brightness-gradient method and colour buys nothing.
    """
    if previous.ndim == 3:
        previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    if current.ndim == 3:
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,  # pyramid scale
        4,  # levels: enough for the largest motions, a swing at 10 Hz
        15,  # window: smaller is sharper but noisier on flat paint
        3,  # iterations
        5,  # polynomial neighbourhood
        1.2,  # polynomial sigma
        0,
    )


def decompose(
    flow: np.ndarray,
    mask: np.ndarray,
    pivot: tuple[float, float],
    scale: float,
    dt: float,
    radial_fraction: float,
    distal_fraction: float,
    erode_pixels: int,
    min_pixels: int,
    moving_threshold_px: float,
) -> tuple[float, float, float, float, float]:
    """Reduce one flow field to the four numbers the state machine can read.

    Args:
        flow: ``(H, W, 2)`` from :func:`dense_flow`.
        mask: the machine's pixels in the *first* of the two frames.
        pivot: the slew centre, in pixels.
        scale: the machine's reach ``L`` in pixels; linear rates are divided by it.
        dt: seconds between the two frames.
        radial_fraction: quantile of radius above which a pixel counts as part
            of the moving arm for ``omega`` and ``v_radial``. The body does not
            rotate about the pivot, so including it only dilutes the estimate.
        distal_fraction: the same idea, stricter, for ``v_up`` -- the bucket is
            at the far end, and its vertical motion is the dig/dump evidence.
        erode_pixels: shrink the mask by this much before sampling, to keep the
            silhouette's edge out of the statistics.
        min_pixels: below this many usable pixels, return NaN rather than a
            number computed from too little.
        moving_threshold_px: displacement below which a pixel is called still,
            used only to report ``coverage``.

    Returns:
        ``(omega, v_radial, v_up, speed, coverage)``; any may be NaN.
    """
    nan = float("nan")
    if mask.sum() < min_pixels:
        return nan, nan, nan, nan, nan

    inner = mask
    if erode_pixels > 0:
        kernel = np.ones((erode_pixels * 2 + 1,) * 2, np.uint8)
        shrunk = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        # Only accept the erosion if it left something to measure: a thin arm
        # can vanish entirely, and no measurement is worse than an edgy one.
        if shrunk.sum() >= min_pixels:
            inner = shrunk

    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width]
    rx = xs - pivot[0]
    ry = ys - pivot[1]
    radius = np.hypot(rx, ry)

    selected = inner & (radius > 1.0)  # the pivot itself has no angular sense
    if selected.sum() < min_pixels:
        return nan, nan, nan, nan, nan

    fx = flow[..., 0]
    fy = flow[..., 1]
    magnitude = np.hypot(fx, fy)
    coverage = float((magnitude[selected] > moving_threshold_px).mean())
    speed = float(np.median(magnitude[selected])) / scale / dt

    def outer(fraction: float) -> np.ndarray:
        cut = np.quantile(radius[selected], fraction)
        return selected & (radius >= cut)

    arm = outer(radial_fraction)
    omega = nan
    v_radial = nan
    if arm.sum() >= min_pixels // 2:
        # Bearing is atan2(-(y-cy), x-cx), so the cross product is taken in that
        # same y-flipped frame: (rx, -ry) x (fx, -fy) = ry*fx - rx*fy.
        cross = ry[arm] * fx[arm] - rx[arm] * fy[arm]
        dot = rx[arm] * fx[arm] + ry[arm] * fy[arm]
        omega = float(np.median(cross / radius[arm] ** 2)) / dt
        v_radial = float(np.median(dot / radius[arm])) / scale / dt

    distal = outer(distal_fraction)
    v_up = nan
    if distal.sum() >= min_pixels // 4:
        # Image y grows downward; negate so that "up" is positive, as elevation is.
        v_up = float(np.median(-fy[distal])) / scale / dt

    return omega, v_radial, v_up, speed, coverage


def build_motion_field(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    pivot: tuple[float, float],
    scale: float,
    dt: float,
    radial_fraction: float = 0.5,
    distal_fraction: float = 0.8,
    erode_pixels: int = 2,
    min_pixels: int = 150,
    moving_threshold_px: float = 0.3,
) -> MotionField:
    """Measure the motion field across a whole sampled sequence.

    ``frames`` and ``masks`` are parallel and in sample order. This is the join
    that has to be got right: masks are stored by sample ordinal, not by source
    frame index, and mixing the two silently pairs each frame with a mask from
    somewhere else in the video. Both lists are indexed the same way here so
    there is nothing to confuse.
    """
    if len(frames) != len(masks):
        raise ValueError(f"{len(frames)} frames but {len(masks)} masks; they must be parallel")

    n = len(frames)
    omega = np.full(n, np.nan)
    v_radial = np.full(n, np.nan)
    v_up = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    coverage = np.full(n, np.nan)

    for index in range(n - 1):
        if frames[index] is None or frames[index + 1] is None or masks[index] is None:
            continue
        flow = dense_flow(frames[index], frames[index + 1])
        (
            omega[index],
            v_radial[index],
            v_up[index],
            speed[index],
            coverage[index],
        ) = decompose(
            flow,
            masks[index],
            pivot,
            scale,
            dt,
            radial_fraction,
            distal_fraction,
            erode_pixels,
            min_pixels,
            moving_threshold_px,
        )

    measured = int(np.isfinite(omega).sum())
    log.info("motion field measured on %d of %d sample intervals", measured, n - 1)
    return MotionField(
        omega=omega, v_radial=v_radial, v_up=v_up, speed=speed, coverage=coverage
    )


def save_motion(field: MotionField, times: np.ndarray, output_dir: str | Path) -> None:
    """Write the motion field beside the features, in both machine and human form."""
    output_dir = Path(output_dir)
    columns = {
        "time_seconds": times,
        "omega": field.omega,
        "v_radial": field.v_radial,
        "v_up": field.v_up,
        "speed": field.speed,
        "coverage": field.coverage,
    }
    np.savez_compressed(output_dir / "motion.npz", **columns)
    rows = np.column_stack([np.asarray(v, dtype=float) for v in columns.values()])
    np.savetxt(
        output_dir / "motion.csv",
        rows,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.6g",
    )


def load_motion(output_dir: str | Path) -> tuple[MotionField, np.ndarray]:
    """Read back what :func:`save_motion` wrote."""
    with np.load(Path(output_dir) / "motion.npz") as data:
        times = data["time_seconds"]
        field = MotionField(
            omega=data["omega"],
            v_radial=data["v_radial"],
            v_up=data["v_up"],
            speed=data["speed"],
            coverage=data["coverage"],
        )
    return field, times


def region_omega(
    flow: np.ndarray,
    region: np.ndarray,
    pivot: tuple[float, float],
    dt: float,
    min_pixels: int = 40,
) -> float:
    """Rotation rate of one region about the pivot, in rad/s.

    The same reduction ``decompose`` performs, exposed so that two regions can
    be compared. ``|r|**2`` in the denominator is what makes that comparison
    mean something: for a rigid body it returns the *same* value at every
    radius, so a difference between two regions is a JOINT rate rather than the
    ``v = omega * r`` velocity gradient you get from comparing raw pixel speeds.

    Measured on the development video, comparing raw velocities made a rigid
    swing look like 6.50 of articulation; comparing ``omega`` brought the same
    frames to 0.360.

    Sign convention follows ``features.bearing`` -- ``atan2(-(y-cy), x-cx)`` --
    so the cross product is taken with the y axis flipped. Returns NaN when the
    region is too small to reduce.
    """
    if region.sum() < min_pixels:
        return float("nan")

    height, width = region.shape
    ys, xs = np.mgrid[0:height, 0:width]
    rx = xs[region] - pivot[0]
    ry = ys[region] - pivot[1]
    radius_squared = np.maximum(rx**2 + ry**2, 1.0)

    fx = flow[..., 0][region]
    fy = flow[..., 1][region]
    cross = ry * fx - rx * fy
    return float(np.median(cross / radius_squared)) / dt
