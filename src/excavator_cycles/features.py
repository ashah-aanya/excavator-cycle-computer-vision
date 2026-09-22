"""Stage 3b: per-sample measurements, from the masks and the scene.

This is the last stage before the state machine, and its output is the table
every phase boundary is read from. Two rules govern it:

**Everything is dimensionless or in seconds.** Distances are divided by the
machine's reach, rates are per second. A hidden video at a different scale or
frame rate produces the same numbers for the same motion.

**Derivatives use a symmetric filter.** A one-sided filter -- a moving average, an
exponential smoother -- delays a signal by an amount that depends on its shape.
That delay would land directly on the phase boundaries, and a boundary that is
consistently late is exactly the systematic error the +/-0.6 s tolerance cannot
absorb. A symmetric filter does not move extrema at all.

The gate for this stage is visual: plot the signals, and check that the four
phase boundaries are visible to a human before asking a rule to find them. If
you cannot see a boundary in the plot, no threshold will find it reliably.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .config import Config
from .geometry import (
    Scene,
    arm_reach,
    bucket_axis,
    bucket_region,
    dwell_clusters,
    farthest_point,
    relative_angle,
    return_direction,
    rotation_centre,
    surface_height,
    unwrap_axis,
)
from .logging_setup import get_logger
from .track import TrackResult

log = get_logger(__name__)


@dataclass
class FeatureTable:
    """One row per sample; every column either dimensionless or in seconds."""

    time_seconds: np.ndarray
    frame_index: np.ndarray
    bearing: np.ndarray  # angle of the bucket about the pivot, unwrapped (rad)
    slew_rate: np.ndarray  # d(bearing)/dt; its SIGN separates hauling from swinging
    extension: np.ndarray  # bucket distance from the pivot, in units of L
    elevation: np.ndarray  # bucket's lowest point above the pivot, in units of L
    elevation_rate: np.ndarray
    speed: np.ndarray  # bucket speed, L per second
    curl: np.ndarray  # bucket long-axis angle (rad) -- the dumping cue
    curl_rate: np.ndarray
    area: np.ndarray  # mask area / L^2; collapses when the bucket is buried
    elongation: np.ndarray  # long/short side of the bucket region; low = axis unreliable
    curl_valid: np.ndarray  # bool: was the bucket axis actually measurable?
    falling: np.ndarray  # downward pixel motion below the bucket -- INDEPENDENT of the mask
    in_dig_zone: np.ndarray  # bool
    in_dump_zone: np.ndarray  # bool
    confidence: np.ndarray  # from the tracker; low means MISSING, not "no"
    valid: np.ndarray  # bool

    def columns(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self.__dict__.items()}

    def __len__(self) -> int:
        return len(self.time_seconds)


def build_features(
    result: TrackResult, masks: dict[int, np.ndarray], config: Config
) -> tuple[FeatureTable, Scene]:
    """Turn masks into the signals the state machine reads."""
    positions = sorted(masks)
    times = np.array([result.frames[p].time_seconds for p in positions])
    frame_indices = np.array([result.frames[p].frame_index for p in positions])
    confidence = np.array([result.frames[p].sam_confidence for p in positions])
    mask_list = [masks[p] for p in positions]

    centre = rotation_centre(mask_list, config.geometry.occupancy_quantile)
    tips = [farthest_point(m, centre) for m in mask_list]
    scale = arm_reach(tips, centre, config.geometry.reach_percentile)
    log.info("pivot at (%.0f, %.0f); arm reach L = %.0f px", centre[0], centre[1], scale)

    xs = np.array([t[0] if t else np.nan for t in tips])
    ys = np.array([t[1] if t else np.nan for t in tips])

    # Image y grows downward, so negate it to make "up" positive.
    bearing = np.unwrap(np.arctan2(-(ys - centre[1]), xs - centre[0]))
    extension = np.hypot(xs - centre[0], ys - centre[1]) / scale

    elevation = np.full(len(positions), np.nan)
    curl = np.full(len(positions), np.nan)
    area = np.full(len(positions), np.nan)
    elongation = np.full(len(positions), np.nan)

    bucket_radius = config.geometry.bucket_radius_frac * scale
    # A second, wider region centred further back gives the forearm's direction.
    # The curl is measured against it: the absolute bucket axis swings through a
    # whole revolution as the arm slews, burying the small rotation that signals
    # dumping.
    forearm_radius = config.geometry.elbow_frac * scale

    for index, (mask, tip) in enumerate(zip(mask_list, tips, strict=True)):
        area[index] = mask.sum() / (scale**2)
        if tip is None:
            continue

        region = bucket_region(mask, tip, bucket_radius)
        if not region.any():
            continue
        # The LOWEST pixel of the bucket, because the task defines hauling as
        # beginning when the *entire* bucket clears the material surface.
        elevation[index] = (centre[1] - np.nonzero(region)[0].max()) / scale

        bucket = bucket_axis(region)
        forearm = bucket_axis(bucket_region(mask, tip, forearm_radius))
        if bucket is None or forearm is None:
            continue
        axis, ratio = bucket
        elongation[index] = ratio
        # A nearly round region has an arbitrary long axis: a couple of pixels
        # flipping swings it by 90 degrees. Reject rather than record noise --
        # a missing measurement is honest, a confident wrong one is not.
        if ratio >= config.geometry.min_bucket_elongation:
            curl[index] = relative_angle(axis, forearm[0])

    curl_valid = np.isfinite(curl)
    curl = unwrap_axis(_fill_gaps(curl))
    if curl_valid.mean() < 0.5:
        log.warning(
            "the bucket axis was measurable on only %.0f%% of samples; the curl "
            "signal is mostly interpolation and should not be used as a clock cue",
            curl_valid.mean() * 100,
        )

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    window = _odd(max(3, round(config.features.smoothing_window_seconds / dt)))
    order = min(config.features.smoothing_polyorder, window - 1)

    bearing_s = _smooth(bearing, window, order)
    elevation_s = _smooth(elevation, window, order)
    curl_s = _smooth(curl, window, order)

    slew_rate = _derivative(bearing_s, window, order, dt)
    elevation_rate = _derivative(elevation_s, window, order, dt)
    curl_rate = _derivative(curl_s, window, order, dt)

    speed = np.hypot(
        _derivative(_smooth(xs / scale, window, order), window, order, dt),
        _derivative(_smooth(ys / scale, window, order), window, order, dt),
    )

    dig_zone, dump_zone, separation = dwell_clusters(tips, speed, config, scale)
    in_dig = _within(xs, ys, dig_zone, config.geometry.min_zone_separation * scale / 2)
    in_dump = _within(xs, ys, dump_zone, config.geometry.min_zone_separation * scale / 2)
    surface = surface_height(elevation_s, in_dig)
    if surface is not None:
        log.info("material surface at %.3f L above the pivot", surface)

    scene = Scene(
        centre=centre,
        scale=scale,
        dig_zone=dig_zone,
        dump_zone=dump_zone,
        surface_height=surface,
        return_sign=return_direction(dig_zone, dump_zone, centre),
        zone_separation=separation,
    )

    falling = falling_material(result, tips, scale, config)

    table = FeatureTable(
        time_seconds=times,
        frame_index=frame_indices,
        bearing=bearing_s,
        slew_rate=slew_rate,
        extension=extension,
        elevation=elevation_s,
        elevation_rate=elevation_rate,
        speed=speed,
        curl=curl_s,
        curl_rate=curl_rate,
        area=area,
        elongation=elongation,
        curl_valid=curl_valid,
        falling=falling,
        in_dig_zone=in_dig,
        in_dump_zone=in_dump,
        confidence=confidence,
        valid=np.isfinite(elevation) & (confidence >= config.features.min_sample_confidence),
    )
    return table, scene


def falling_material(
    result: TrackResult, tips: list[tuple[float, float] | None], scale: float, config: Config
) -> np.ndarray:
    """Downward pixel motion just below the bucket: material being released.

    This is the only cue in the pipeline that touches no mask. Every other
    signal is derived from SAM's output, so they share a failure mode -- if the
    mask is wrong they are wrong together, and their agreement proves nothing.
    Optical flow measures the image directly, which is what makes it worth the
    extra decode pass.

    It is corroboration, not a clock: material takes time to fall, so its onset
    lags the tipping that causes it. Using it to time the boundary would bias
    dumping late by however long that lag is.
    """
    import cv2

    from .video import iter_samples

    frames = [
        cv2.cvtColor(s.image, cv2.COLOR_BGR2GRAY)
        for s in iter_samples(result.video, rate_hz=result.rate_hz)
    ]
    if len(frames) != len(tips):
        log.warning(
            "decoded %d frames but have %d samples; skipping the falling-material cue",
            len(frames),
            len(tips),
        )
        return np.full(len(tips), np.nan)

    box = config.geometry.bucket_radius_frac * scale
    signal = np.full(len(tips), np.nan)
    for index in range(1, len(frames)):
        tip = tips[index]
        if tip is None:
            continue
        # A patch directly beneath the bucket, where released material falls.
        x0 = int(max(0, tip[0] - box))
        x1 = int(min(frames[index].shape[1], tip[0] + box))
        y0 = int(max(0, tip[1]))
        y1 = int(min(frames[index].shape[0], tip[1] + 2 * box))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        flow = cv2.calcOpticalFlowFarneback(
            frames[index - 1][y0:y1, x0:x1],
            frames[index][y0:y1, x0:x1],
            None,
            0.5,
            2,
            9,
            2,
            5,
            1.1,
            0,
        )
        # Subtract the bucket's own vertical motion before calling anything
        # "falling". A descending bucket drags its whole neighbourhood downward
        # in the flow field, and without this correction the cue fires hardest
        # while the machine is lowering the bucket to dig -- the opposite of
        # dumping. What is left is motion that outruns the bucket: material.
        previous = tips[index - 1]
        bucket_dy = (tip[1] - previous[1]) if previous is not None else 0.0
        relative = flow[..., 1] - bucket_dy
        signal[index] = float(np.clip(relative, 0, None).mean() / scale)
    return signal


def _smooth(values: np.ndarray, window: int, order: int) -> np.ndarray:
    """Symmetric (zero-phase) smoothing. See the module docstring for why."""
    filled = _fill_gaps(values)
    if np.isnan(filled).all() or len(filled) <= window:
        return filled
    return savgol_filter(filled, window, order)


def _derivative(values: np.ndarray, window: int, order: int, dt: float) -> np.ndarray:
    """Rate of change, also zero-phase, so extrema keep their position in time."""
    filled = _fill_gaps(values)
    if np.isnan(filled).all() or len(filled) <= window:
        return np.full_like(filled, np.nan)
    return savgol_filter(filled, window, order, deriv=1, delta=dt)


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Interpolate short gaps so a filter can run; genuinely absent data stays NaN.

    A gap is a missing measurement, not a measurement of zero, and the `valid`
    column records where that happened so downstream rules can treat those
    samples as absent rather than as evidence.
    """
    out = values.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    indices = np.arange(len(out))
    out[~finite] = np.interp(indices[~finite], indices[finite], out[finite])
    return out


def _within(xs, ys, zone, radius: float) -> np.ndarray:
    if zone is None:
        return np.zeros(len(xs), dtype=bool)
    return np.hypot(xs - zone[0], ys - zone[1]) <= radius


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def save(table: FeatureTable, scene: Scene, output_dir: str | Path) -> None:
    """Write the table and the scene for the state machine and for inspection."""
    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / "features.npz", **table.columns())
    (output_dir / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2))

    # A CSV as well: the point of this stage is that a human can read it.
    columns = table.columns()
    header = ",".join(columns)
    rows = np.column_stack([np.asarray(v, dtype=float) for v in columns.values()])
    np.savetxt(
        output_dir / "features.csv",
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6g",
    )


def load(output_dir: str | Path) -> tuple[FeatureTable, dict[str, Any]]:
    output_dir = Path(output_dir)
    with np.load(output_dir / "features.npz") as data:
        table = FeatureTable(**{k: data[k] for k in data.files})
    scene = json.loads((output_dir / "scene.json").read_text())
    return table, scene
