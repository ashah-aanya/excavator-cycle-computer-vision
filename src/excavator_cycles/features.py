"""Kinematic features from bounding boxes. Design diagram stage 2.

What this computes, and why each one is here
--------------------------------------------
The diagram asks for four groups. Three are built; the fourth, cabin rotation
omega(t), is deliberately absent -- see "No rotation signal" below.

* **Positional** -- the bucket's x and y trajectories, ``dh/dt`` and ``dx/dt``.
* **Relative position** -- bucket against the cabin, and bucket against the truck.
* **Bucket / truck overlap** -- how much of the bucket lies over the truck bed.
* **Shape** -- the bucket box's aspect ratio, which carries its tipping.

Every length is divided by ``L``, the machine's arm reach in pixels, so a feature
means the same thing on a 480-wide video and a 1920-wide one. Every rate is per
second, taken against the decoder's real timestamps rather than a frame index, so
a hidden video at 25 fps needs no special handling.

Two sign conventions, stated once
---------------------------------
Image ``y`` increases **downward**. Getting this wrong inverts the
digging-to-hauling trigger, so the flip happens in exactly two places and both
are named:

* ``height`` is ``(pivot_y - bucket_y) / L`` -- **positive is up**, measured from
  the slew centre. Its derivatives ``dh_dt`` and ``d2h_dt2`` inherit that.
* ``rel_cabin_y`` is ``(cabin_y - bucket_y) / L`` -- **positive means the bucket
  is above the cabin**.

``bucket_y`` itself is left in image coordinates (down is positive) because it is
a trajectory for plotting, not a physics term. Nothing downstream should
differentiate it; differentiate ``height`` instead. There is intentionally no
``dy_dt`` column, because having both it and ``dh_dt`` would be an invitation to
pick the wrong sign.

No rotation signal
------------------
Earlier versions measured the cabin's rotation two ways and both were wrong. The
fitted arm chain gave "rigid" links varying 5.2-8.2x in length. Dense optical flow
over the house gave a quantity that correlated with the real slew at r = 0.025 --
chance -- because a machine slewing about a **vertical** axis translates sideways
in projection rather than rotating in the image plane.

The swing is therefore measured positionally, as ``|dx/dt|`` of the bucket box
centre, which is what the verified detector used.

Nothing here is tied to one video: the frame rate, the truck box, the pivot and
``L`` are all derived from the run being processed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .boxes import BoxTrack, raw_boxes, smooth_boxes
from .config import Config
from .geometry import arm_reach, farthest_point, rotation_centre, stable_core
from .logging_setup import get_logger
from .rates import derivative
from .track import TrackResult

log = get_logger(__name__)


@dataclass(frozen=True)
class Scene:
    """Landmarks derived once per video, shared by every sample.

    All three are measured from this video's own masks and detections. None is a
    constant, and none survives from one video to the next.
    """

    pivot: tuple[float, float]  # slew centre, pixels
    scale: float  # L: arm reach, pixels
    truck_box: tuple[float, float, float, float] | None  # pixels; None = no truck

    def to_dict(self) -> dict[str, Any]:
        return {
            "pivot": list(self.pivot),
            "scale": self.scale,
            "truck_box": list(self.truck_box) if self.truck_box else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        truck = data.get("truck_box")
        return cls(
            pivot=(float(data["pivot"][0]), float(data["pivot"][1])),
            scale=float(data["scale"]),
            truck_box=tuple(float(v) for v in truck) if truck else None,
        )


@dataclass(frozen=True)
class FeatureTable:
    """One row per sample. Lengths in ``L``, rates in ``L``/second."""

    time_seconds: np.ndarray

    # --- positional: the bucket's trajectory (diagram: "x and y trajectories")
    bucket_x: np.ndarray  # box centre x / L. Image coords.
    bucket_y: np.ndarray  # box centre y / L. Image coords: DOWN is positive.
    height: np.ndarray  # (pivot_y - bucket_y) / L. UP is positive.

    # --- positional: the rates the diagram names
    dh_dt: np.ndarray  # L/s, up-positive. Carries T1 and T2.
    d2h_dt2: np.ndarray  # L/s^2. The kink where lifting takes over from scooping.
    dx_dt: np.ndarray  # L/s, right-positive.
    speed_x: np.ndarray  # |dx/dt|. Carries T4.

    # --- relative position (diagram: "relative to truck", "relative to cabin")
    rel_cabin_x: np.ndarray  # (bucket_x - cabin_x) / L
    rel_cabin_y: np.ndarray  # (cabin_y - bucket_y) / L. Positive = bucket above.
    rel_truck_x: np.ndarray  # (bucket_x - truck_centre_x) / L. nan with no truck.
    rel_truck_y: np.ndarray  # (truck_centre_y - bucket_y) / L. Positive = above.

    # --- bucket / truck overlap (diagram: its own box)
    truck_overlap: np.ndarray  # [0,1]: share of the bucket box over the truck box

    # --- shape
    aspect_ratio: np.ndarray  # bucket box width / height. Carries T3.
    radius: np.ndarray  # |bucket - pivot| / L. Distance from the body
    # centroid, which sits low among the tracks -- a proxy for reach, not reach.

    # --- the smoothed boxes themselves, in PIXELS, for drawing and overlap.
    # The only columns here not divided by L: the renderer works in image space,
    # and the annotated video has to show the boxes the pipeline actually used.
    bucket_box: np.ndarray  # (n, 4) x0 y0 x1 y1
    cabin_box: np.ndarray  # (n, 4) x0 y0 x1 y1

    found: np.ndarray  # bool: did this sample have a bucket mask?

    def __len__(self) -> int:
        return len(self.time_seconds)

    def columns(self) -> dict[str, np.ndarray]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def build_features(
    result: TrackResult,
    objects: dict[str, dict[int, np.ndarray]],
    config: Config,
) -> tuple[FeatureTable, Scene]:
    """Derive the scene, then every feature, from the tracked masks.

    Args:
        result: a completed ``track()`` run; supplies the real timestamps.
        objects: mask sets by name. ``"excavator"`` is required, ``"bucket"``
            strongly preferred -- without it the bucket falls back to the whole
            machine's box, which is a much weaker signal and is logged loudly.
    """
    times = np.array([record.time_seconds for record in result.frames], dtype=float)
    count = len(times)
    if count < 2:
        raise ValueError(f"need at least 2 samples to differentiate, got {count}")
    _check_uniform_sampling(times)

    excavator = objects.get("excavator")
    if not excavator:
        raise ValueError("no excavator masks; cannot derive the scene")
    scene = derive_scene(result, excavator, config)

    bucket = objects.get("bucket")
    if not bucket:
        log.warning(
            "no bucket masks in this run; falling back to the whole machine's box. "
            "Every feature below is then about the silhouette, not the bucket."
        )
        bucket = excavator

    window = config.features.box_window_seconds
    alignment = config.features.box_alignment

    # A low-confidence mask is a GAP IN PERCEPTION, not a measurement. Treating one
    # as real lets a bad mask set a box, and every rate derived from that box then
    # carries an excursion the machine never made. `min_sample_confidence` has
    # documented this from the start and nothing read it: the samples were kept
    # whatever SAM 2 thought of them.
    #
    # Dropped here rather than in `track`, so the confidence stays on the record and
    # the threshold remains a question for this stage to answer -- re-running
    # `features` with a different threshold costs a second and needs no GPU.
    raw = raw_boxes(bucket, count)
    floor = config.features.min_sample_confidence
    weak = np.array(
        [
            record.has_bucket_mask and record.bucket_confidence < floor
            for record in result.frames
        ]
    )
    if weak.any():
        log.info(
            "%d of %d bucket masks are below the %.2f confidence floor; treating them as "
            "missing rather than as evidence",
            int(weak.sum()),
            count,
            floor,
        )
        raw = raw.copy()
        raw[weak] = np.nan

    bucket_track = smooth_boxes(raw, times, window, alignment)
    cabin_track = smooth_boxes(
        raw_boxes(_body_masks(excavator, count, config), count), times, window, alignment
    )
    return _assemble(bucket_track, cabin_track, scene, times, config), scene


def _check_uniform_sampling(times: np.ndarray, tolerance: float = 0.05) -> None:
    """Warn when the samples are not evenly spaced in time.

    Every derivative below passes a single scalar spacing to Savitzky-Golay, so
    it assumes a uniform clock. That holds for any constant-frame-rate source,
    because sampling takes an integer frame stride and therefore lands on evenly
    spaced frames. It does NOT hold for variable-frame-rate footage -- phone or
    web video -- where the decoder's timestamps genuinely jump around, and there
    the rates are wrong by whatever the local spacing ratio happens to be,
    silently.

    A warning rather than an error: the answer degrades smoothly rather than
    becoming nonsense, and refusing to run at all would be worse.
    """
    gaps = np.diff(times)
    median = float(np.median(gaps))
    if median <= 0:
        raise ValueError(f"timestamps are not increasing (median spacing {median})")
    spread = float(np.ptp(gaps)) / median
    if spread > tolerance:
        log.warning(
            "sampling is not uniform: intervals span %.1f%% of the median (%.4f s). "
            "Every rate assumes one spacing, so a variable-frame-rate source will "
            "bias them by the local ratio.",
            spread * 100,
            median,
        )


def derive_scene(
    result: TrackResult,
    excavator: dict[int, np.ndarray],
    config: Config,
) -> Scene:
    """Pivot, scale and truck box -- each measured from this video."""
    masks = [excavator[k] for k in sorted(excavator)]
    pivot = rotation_centre(masks, config.geometry.occupancy_quantile)
    scale = arm_reach(
        [farthest_point(m, pivot) for m in masks], pivot, config.geometry.reach_percentile
    )
    truck = truck_box(result, config)
    log.info(
        "scene: pivot (%.0f, %.0f) px, L = %.0f px, truck %s",
        pivot[0],
        pivot[1],
        scale,
        "found" if truck else "none",
    )
    return Scene(pivot=pivot, scale=float(scale), truck_box=truck)


def truck_box(result: TrackResult, config: Config) -> tuple[float, ...] | None:
    """Where the truck is, from the frames where it was cleanly separate.

    The detector sometimes merges the excavator and the truck into one box; on
    those frames the "truck" box is the merged box and says nothing. Frames whose
    two boxes barely overlap are the informative ones, and a parked truck holds
    its position while the excavator swings over it -- so the median of the clean
    detections is a static box good for the whole clip.

    ``None`` when no truck is ever cleanly seen, which is legitimate: a video
    with no truck simply has no overlap feature.
    """
    clean = [
        record.truck_box
        for record in result.frames
        if record.truck_box is not None
        and (record.detection_truck_iou or 0.0) <= config.track.max_clean_truck_iou
    ]
    if not clean:
        return None
    box = np.median(np.array(clean, dtype=float), axis=0)
    return tuple(float(v) for v in box)


def overlap_fraction(
    box: tuple[float, float, float, float],
    other: tuple[float, ...],
) -> float:
    """Share of ``box``'s area that lies inside ``other``.

    Normalised by the bucket rather than by the union, because the question the
    dumping cue asks is "how much of the bucket is over the bed", and a union
    would shrink that answer as the truck grew.

    Boxes here are **inclusive**: ``box_of`` returns ``xs.max()``, so a box from
    x0 to x1 covers ``x1 - x0 + 1`` pixel columns. Every term below carries that
    ``+ 1``. Dropping it is not a rounding difference -- a bucket box one pixel
    tall sitting entirely on the bed would compute ``0 * n / 0`` and report
    **no overlap at all**, and the bucket is thinnest in projection at exactly
    the moment it tips over to dump, which is the event this gates.
    """
    wide = max(0.0, min(box[2], other[2]) - max(box[0], other[0]) + 1.0)
    tall = max(0.0, min(box[3], other[3]) - max(box[1], other[1]) + 1.0)
    area = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)  # >= 1 by construction
    return float(wide * tall / area)


def _body_masks(
    excavator: dict[int, np.ndarray], count: int, config: Config
) -> list[np.ndarray | None]:
    """The cabin, per frame: this frame's mask restricted to the persistent core.

    The core is the set of pixels that are machine in almost every frame -- the
    body and undercarriage, which stay put while the arm sweeps. Intersecting it
    with the current mask gives a cabin box that does not grow when the arm
    extends, which matters because three phase cues compare the bucket against
    the cabin. A reference that drifted with the arm would compare the bucket
    against something the bucket itself had dragged.
    """
    masks = [excavator[k] for k in sorted(excavator)]
    core, _threshold = stable_core(masks, config.geometry.occupancy_quantile)
    out: list[np.ndarray | None] = []
    for index in range(count):
        mask = excavator.get(index)
        out.append(None if mask is None else (mask & core))
    return out


def _assemble(
    bucket: BoxTrack,
    cabin: BoxTrack,
    scene: Scene,
    times: np.ndarray,
    config: Config,
) -> FeatureTable:
    """Every column, from the two smoothed box tracks."""
    scale = scene.scale
    window = config.features.derivative_window_seconds
    order = config.features.smoothing_polyorder

    bx = bucket.centre_x / scale
    by = bucket.centre_y / scale
    cx = cabin.centre_x / scale
    cy = cabin.centre_y / scale

    # The one y-flip that matters: up-positive height above the slew centre.
    height = (scene.pivot[1] / scale) - by

    dh = derivative(height, times, window_seconds=window, polyorder=order)
    dx = derivative(bx, times, window_seconds=window, polyorder=order)

    if scene.truck_box is not None:
        tx = (scene.truck_box[0] + scene.truck_box[2]) / 2.0 / scale
        ty = (scene.truck_box[1] + scene.truck_box[3]) / 2.0 / scale
        rel_truck_x, rel_truck_y = bx - tx, ty - by
        overlap = np.array(
            [
                overlap_fraction(
                    (bucket.x0[i], bucket.y0[i], bucket.x1[i], bucket.y1[i]),
                    scene.truck_box,
                )
                if bucket.found[i]
                else np.nan
                for i in range(len(bucket))
            ]
        )
    else:
        rel_truck_x = np.full(len(bucket), np.nan)
        rel_truck_y = np.full(len(bucket), np.nan)
        overlap = np.full(len(bucket), np.nan)

    return FeatureTable(
        time_seconds=times,
        bucket_x=bx,
        bucket_y=by,
        height=height,
        dh_dt=dh,
        d2h_dt2=derivative(dh, times, window_seconds=window, polyorder=order),
        dx_dt=dx,
        speed_x=np.abs(dx),
        rel_cabin_x=bx - cx,
        rel_cabin_y=cy - by,
        rel_truck_x=rel_truck_x,
        rel_truck_y=rel_truck_y,
        truck_overlap=overlap,
        aspect_ratio=bucket.aspect_ratio,
        radius=np.hypot(bx - scene.pivot[0] / scale, by - scene.pivot[1] / scale),
        bucket_box=np.stack([bucket.x0, bucket.y0, bucket.x1, bucket.y1], axis=1),
        cabin_box=np.stack([cabin.x0, cabin.y0, cabin.x1, cabin.y1], axis=1),
        found=bucket.found,
    )


def save(table: FeatureTable, scene: Scene, output_dir: str | Path) -> None:
    """Write ``features.npz`` and ``scene.json`` beside the masks."""
    output_dir = Path(output_dir)
    np.savez_compressed(output_dir / "features.npz", **table.columns())
    (output_dir / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2))
    log.info("wrote features.npz and scene.json to %s", output_dir)


def load(output_dir: str | Path) -> tuple[FeatureTable, Scene]:
    """Read back what :func:`save` wrote."""
    output_dir = Path(output_dir)
    data = np.load(output_dir / "features.npz")
    table = FeatureTable(**{f: data[f] for f in FeatureTable.__dataclass_fields__})
    scene = Scene.from_dict(json.loads((output_dir / "scene.json").read_text()))
    return table, scene
