"""The four kinematic signals the state machine reads.

What the spec actually asks for
-------------------------------
Only four **onsets** are defined; each phase ends where the next begins. So the
job is to find four moments, never to classify a frame. Three of the four need
the bucket separated from the rest of the machine:

===  ==========  =========================================================
T1   digging     ``h`` arrives at a plateau, and the scoop begins
T2   hauling     ``h`` departs that plateau
T3   dumping     ``delta_omega`` departs zero, WHILE over the truck
T4   swinging    ``omega_house`` departs zero
===  ==========  =========================================================

Why these four and not others
-----------------------------
Every other candidate was measured and dropped. Separability -- the spread of
phase medians over the within-phase scatter -- on the development video:

    mask area           0.93    the whole machine barely changes shape
    falling material    0.85    and its PEAK is during hauling, not dumping
    radial velocity     0.82    no phase signal at all
    chain speed         1.25    superseded

Anything under about 2 cannot tell phases apart.

Two things worth knowing before reading the code
------------------------------------------------
**Rates come from flow, levels come from geometry.** Optical flow reduces
thousands of vectors with a median and never makes a hard choice, so it measures
rates well; it cannot measure a level. Levels come from the mask, but only as
aggregates -- ``h`` is a high percentile of the bucket's rows, never an argmax,
because an extreme point flips and a percentile degrades.

**``h`` does not need a calibrated ground level.** While the bucket is buried,
its lowest *visible* row is where it enters the dirt, so ``h`` pins -- measured
at +/-0.0005 L for the better part of a second. T1 and T2 are therefore
*departures from h's own plateau*, not crossings of a surface height that would
have to be estimated. That removes a whole source of differential bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .motion import dense_flow, region_omega
from .seeding import geodesic_bands

log = logging.getLogger(__name__)


@dataclass
class KinematicSignals:
    """One row per sample. Rates are rad/s; ``h`` is in units of ``L``."""

    time_seconds: np.ndarray
    omega_house: np.ndarray
    omega_stick: np.ndarray
    omega_bucket: np.ndarray
    delta_omega: np.ndarray
    h: np.ndarray
    truck_overlap: np.ndarray
    valid: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        """Drives both persistence formats; a new field flows through for free."""
        return dict(self.__dict__)

    def __len__(self) -> int:
        return len(self.time_seconds)


def bucket_height(bucket: np.ndarray, pivot: tuple[float, float], scale: float) -> float:
    """How low the bucket reaches, relative to the pivot, in units of ``L``.

    A high percentile of the bucket's rows rather than its lowest pixel. The
    lowest pixel is an argmax over a boundary, and boundaries deform by 10-30 px
    whenever parts overlap; a percentile trims that off and still tracks the
    bottom of the bucket. Positive is up, matching ``features.elevation``.
    """
    if not bucket.any():
        return float("nan")
    rows = np.nonzero(bucket)[0]
    lowest = float(np.percentile(rows, 95))
    return (pivot[1] - lowest) / scale


def truck_overlap(bucket: np.ndarray, truck_box: tuple[float, float, float, float]) -> float:
    """Fraction of the bucket sitting inside the truck's bed box.

    The gate for T3, and half of what the spec's definition demands: dumping
    begins when the bucket *reaches the dumping location* **and** starts to tip.
    Either half alone is wrong -- at 14.8 s on the development video the bucket
    visibly tilts and sheds material 3.8 s before the real dump, while still in
    transport, and only the location half separates the two.

    The truck box is free: it is static across all 296 samples, spread
    ``[0, 0, 0, 0]``, because a parked truck does not move.
    """
    if not bucket.any():
        return float("nan")
    x0, y0, x1, y1 = truck_box
    ys, xs = np.nonzero(bucket)
    inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return float(inside.mean())


def build_signals(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    core: np.ndarray,
    pivot: tuple[float, float],
    scale: float,
    dt: float,
    truck_box: tuple[float, float, float, float] | None = None,
    times: np.ndarray | None = None,
    house_fraction: float = 0.45,
    min_pixels: int = 40,
) -> KinematicSignals:
    """Measure all four signals over a clip.

    ``frames`` and ``masks`` must be **parallel lists in sample order**. Masks on
    disk are keyed by sample ordinal while ``track.json`` carries source frame
    indices; joining on the wrong one pairs each frame with a mask from
    elsewhere in the video and produces a confident null result. That has
    happened once on this project, so the lengths are checked here.

    The last entry of every rate is NaN: ``n`` samples give ``n - 1`` intervals.
    """
    if len(frames) != len(masks):
        raise ValueError(
            f"{len(frames)} frames but {len(masks)} masks; they must be parallel"
        )

    n = len(frames)
    blank = lambda: np.full(n, np.nan)  # noqa: E731
    omega_house, omega_stick, omega_bucket = blank(), blank(), blank()
    height, overlap = blank(), blank()

    ys, xs = np.mgrid[0 : masks[0].shape[0], 0 : masks[0].shape[1]] if n else (None, None)
    radius = np.hypot(xs - pivot[0], ys - pivot[1]) if n else None

    for index in range(n):
        mask = masks[index]
        if mask is None or not mask.any():
            continue

        bucket, stick, _ = geodesic_bands(mask, core, pivot)
        height[index] = bucket_height(bucket, pivot, scale)
        if truck_box is not None:
            overlap[index] = truck_overlap(bucket, truck_box)

        if index >= n - 1:
            continue  # no interval after the last sample
        flow = dense_flow(frames[index], frames[index + 1])
        house = mask & (radius < house_fraction * radius[mask].max())
        omega_house[index] = region_omega(flow, house, pivot, dt, min_pixels)
        omega_stick[index] = region_omega(flow, stick, pivot, dt, min_pixels)
        omega_bucket[index] = region_omega(flow, bucket, pivot, dt, min_pixels)

    delta = omega_bucket - omega_stick
    valid = np.isfinite(height)

    log.info(
        "signals measured: h on %d/%d samples, delta_omega on %d/%d intervals",
        int(np.isfinite(height).sum()),
        n,
        int(np.isfinite(delta).sum()),
        max(n - 1, 0),
    )
    return KinematicSignals(
        time_seconds=(np.arange(n) * dt if times is None else np.asarray(times)),
        omega_house=omega_house,
        omega_stick=omega_stick,
        omega_bucket=omega_bucket,
        delta_omega=delta,
        h=height,
        truck_overlap=overlap,
        valid=valid,
    )


def save_signals(signals: KinematicSignals, output_dir: str | Path) -> None:
    """Write ``signals.npz`` and ``signals.csv`` side by side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = signals.columns()

    np.savez_compressed(output_dir / "signals.npz", **columns)
    table = np.column_stack([np.asarray(v, dtype=np.float64) for v in columns.values()])
    np.savetxt(
        output_dir / "signals.csv",
        table,
        delimiter=",",
        header=",".join(columns),
        comments="",
        fmt="%.6g",
    )
    log.info("wrote %s", output_dir / "signals.csv")


def load_signals(output_dir: str | Path) -> KinematicSignals:
    """Read back what ``save_signals`` wrote."""
    data: Any = np.load(Path(output_dir) / "signals.npz")
    return KinematicSignals(**{key: data[key] for key in data.files})
