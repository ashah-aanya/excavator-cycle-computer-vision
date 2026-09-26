"""The cabin reference box, and a diagnostic video that shows how stable it is.

Why this exists
---------------
Several finite-state-machine cues in the v2 design are comparisons against the
cabin: "cabin is to the left of the bucket", "higher x value and lower y value
than cabin". None of them needs the cabin located precisely -- they need a
reference that does not *move while the bucket moves*. A reference that drifts
with the arm would compare the bucket against something the bucket itself
dragged, which is a silent, plausible-looking error.

How the cabin is found
----------------------
No detector is involved. An excavator's body and undercarriage stay put while
the arm sweeps, so *persistence* separates the two on its own:

    occupancy[y, x] = fraction of sampled frames in which this pixel was machine

Like a long-exposure photograph of a ceiling fan: the hub stays sharp because
something is always there, the blades smear because any one point on the disc
is only covered part of the time. Pixels with high occupancy are body; pixels
the arm sweeps through have low occupancy.

The cutoff is a *quantile of that video's own occupancy distribution*, never a
pixel value, so it re-derives itself per video (``geometry.occupancy_quantile``).

Why a box rather than a point
-----------------------------
A centroid is the more stable single number, but it throws away extent, and
extent is what a "has the bucket moved past the cabin" test actually wants. The
edges are not equally trustworthy, and the asymmetry is physical: the machine's
rear and underside end at a hard silhouette boundary, while the front and top
blend into the boom, where occupancy is graded rather than binary.

Measured on the development clip, sweeping the quantile over 0.85-0.95 (which
changes the core's area by 2.2x):

    right edge   +/- 3 px     <- hard boundary
    bottom edge  +/- 1 px     <- hard boundary
    left edge    +/- 32 px    <- blends into the boom
    top edge     +/- 29 px    <- blends into the boom

So cues should be written against the right and bottom edges. A cue that leans
on the left or top edge is resting on the soft side of the silhouette and will
not survive a different machine pose. :func:`stability` recomputes this table
for whatever video it is given, rather than trusting the numbers above.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get_logger
from .masks import load as load_masks
from .track import load_result
from .video import probe

log = get_logger(__name__)

# Cosmetic only, and deliberately not in the config: these choose how the
# diagnostic looks, never what it measures. Same rationale as render.py.
_STATIC_COLOR = (90, 220, 250)  # the box the pipeline would actually use
_ROLLING_COLOR = (250, 140, 90)  # recomputed from a sliding window
_CORE_COLOR = (80, 230, 90)
_TEXT = (255, 255, 255)
_PANEL = (28, 28, 28)
_CORE_ALPHA = 0.35


Box = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class CabinBox:
    """The cabin reference, plus the evidence for how it was chosen."""

    box: Box
    centroid: tuple[float, float]
    threshold: float  # the occupancy cutoff this video produced
    core_pixels: int
    components: int  # connected components in the core; 1 means no specks

    @property
    def right(self) -> float:
        """The trustworthy edge: use this for 'past the cabin' comparisons."""
        return self.box[2]

    @property
    def bottom(self) -> float:
        """The other trustworthy edge."""
        return self.box[3]


def occupancy(masks: list[np.ndarray]) -> np.ndarray:
    """Per-pixel fraction of frames in which that pixel was machine.

    The long exposure. Float in [0, 1], same shape as one mask.
    """
    if not masks:
        raise ValueError("no masks given; cannot compute occupancy")
    return np.mean(np.stack(masks).astype(np.float32), axis=0)


def stable_core(masks: list[np.ndarray], quantile: float) -> tuple[np.ndarray, float]:
    """The pixels that are machine in almost every frame -- the body.

    Args:
        masks: one boolean mask per sample.
        quantile: position in *this video's* occupancy distribution, not a
            pixel value and not an occupancy level. 0.90 means "keep the most
            persistent tenth of the pixels that were ever machine".

    Returns:
        The boolean core mask, and the occupancy cutoff it worked out to. The
        cutoff is returned so callers can report what the quantile actually
        meant on this video, which differs between videos by design.
    """
    occ = occupancy(masks)
    ever = occ[occ > 0]
    if ever.size == 0:
        raise ValueError("masks are empty; no machine pixels anywhere")

    threshold = float(np.quantile(ever, quantile))
    core = occ >= max(threshold, np.finfo(np.float32).tiny)
    if not core.any():
        # Degenerate only if every pixel shares one occupancy value; fall back
        # to "was ever machine" rather than returning an empty core.
        log.warning("occupancy quantile %.2f produced an empty core; using all", quantile)
        core = occ > 0
    return core, threshold


def cabin_box(masks: list[np.ndarray], quantile: float) -> CabinBox:
    """Fit a box to the stable core.

    The core is expected to be a single connected blob -- on the development
    clip it is, at every quantile tried. ``components`` is reported rather than
    silently cleaned up: more than one means the body broke apart, which is a
    perception problem worth seeing rather than papering over with a
    largest-component filter.
    """
    core, threshold = stable_core(masks, quantile)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), connectivity=8
    )
    components = count - 1  # label 0 is the background
    if components > 1:
        areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
        log.warning("cabin core split into %d components (areas %s)", components, areas[:4])

    ys, xs = np.nonzero(core)
    return CabinBox(
        box=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        centroid=(float(xs.mean()), float(ys.mean())),
        threshold=threshold,
        core_pixels=int(core.sum()),
        components=components,
    )


def stability(
    masks: list[np.ndarray],
    quantile: float,
    times_seconds: list[float] | None = None,
    window_seconds: float = 3.0,
) -> dict[str, object]:
    """Measure how much the box moves, two independent ways.

    **Quantile sensitivity** re-fits the box across a sweep of quantiles. This
    answers "how much does the box depend on where I drew the body/arm line?"

    **Temporal sensitivity** re-fits the box from a sliding window of frames.
    This answers "would the box be in the same place if I had only seen part of
    the video?", which is the property the FSM cues actually rely on.

    Args:
        window_seconds: length of the sliding window. Diagnostic only -- it
            sets what this report measures, never what the pipeline computes.
            In seconds so it means the same thing at any frame rate.
    """
    report: dict[str, object] = {}

    # --- how much does the box depend on the quantile? ---
    sweep = {}
    for q in (0.80, 0.85, 0.90, 0.95):
        fitted = cabin_box(masks, q)
        sweep[q] = fitted.box
    edges = np.array(list(sweep.values()))  # rows: quantiles, cols: x0 y0 x1 y1
    report["quantile_sweep"] = sweep
    report["quantile_edge_range"] = {
        "left": float(np.ptp(edges[:, 0])),
        "top": float(np.ptp(edges[:, 1])),
        "right": float(np.ptp(edges[:, 2])),
        "bottom": float(np.ptp(edges[:, 3])),
    }

    # --- how much does the box depend on WHICH frames it saw? ---
    if times_seconds is None:
        times_seconds = list(range(len(masks)))
    spacing = float(np.median(np.diff(times_seconds))) if len(times_seconds) > 1 else 1.0
    window = max(2, round(window_seconds / spacing))

    rolling: list[Box] = []
    for start in range(0, len(masks) - window + 1):
        chunk = masks[start : start + window]
        try:
            rolling.append(cabin_box(chunk, quantile).box)
        except ValueError:
            continue

    if rolling:
        arr = np.array(rolling)
        report["rolling_window_seconds"] = window_seconds
        report["rolling_window_samples"] = window
        report["rolling_count"] = len(rolling)
        report["rolling_edge_std"] = {
            "left": float(arr[:, 0].std()),
            "top": float(arr[:, 1].std()),
            "right": float(arr[:, 2].std()),
            "bottom": float(arr[:, 3].std()),
        }
        report["rolling_edge_range"] = {
            "left": float(np.ptp(arr[:, 0])),
            "top": float(np.ptp(arr[:, 1])),
            "right": float(np.ptp(arr[:, 2])),
            "bottom": float(np.ptp(arr[:, 3])),
        }
    return report


def rolling_boxes(
    masks: list[np.ndarray],
    quantile: float,
    window: int,
) -> list[Box | None]:
    """A box per sample, each fitted from the ``window`` samples around it.

    Centred, not trailing, so the box does not lag the footage it describes.
    Entries are ``None`` where the window would run off either end.
    """
    half = window // 2
    out: list[Box | None] = []
    for index in range(len(masks)):
        start, stop = index - half, index + half + 1
        if start < 0 or stop > len(masks):
            out.append(None)
            continue
        try:
            out.append(cabin_box(masks[start:stop], quantile).box)
        except ValueError:
            out.append(None)
    return out


def render_diagnostic(
    track_dir: str | Path,
    quantile: float,
    out_path: str | Path | None = None,
    scale: float = 2.0,
    window_seconds: float = 3.0,
) -> dict[str, object]:
    """Write a video showing the cabin box, so its stability can be watched.

    Draws two boxes on every frame:

    * the **static** box, fitted once from the whole clip -- what the pipeline
      would actually use as the cabin reference; and
    * the **rolling** box, refitted from a window centred on the current frame.

    The static box cannot move by construction. The rolling box is the thing to
    watch: wherever it sits still, the estimate does not depend on which frames
    it saw, which is exactly the stability the FSM cues need. Wherever it jumps,
    a cue written against that edge would be fragile.
    """
    track_dir = Path(track_dir)
    result, _masks_unused = load_result(track_dir)
    by_frame, _shape = load_masks(track_dir / "masks.npz")
    ordered_keys = sorted(by_frame)
    masks = [by_frame[key].astype(bool) for key in ordered_keys]

    out_path = Path(out_path) if out_path else track_dir / "cabin_box.mp4"

    times = [record.time_seconds for record in result.frames]
    static = cabin_box(masks, quantile)
    report = stability(masks, quantile, times, window_seconds)
    spacing = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    window = max(2, round(window_seconds / spacing))
    rolling = rolling_boxes(masks, quantile, window)
    core, _threshold = stable_core(masks, quantile)

    log.info(
        "cabin box %s | core %d px, %d component(s), occupancy cutoff %.2f",
        tuple(round(v, 1) for v in static.box),
        static.core_pixels,
        static.components,
        static.threshold,
    )

    info = probe(result.video, verify=True)
    width, height = round(result.width * scale), round(result.height * scale)
    panel = max(56, round(height * 0.20))

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (width, height + panel)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    capture = cv2.VideoCapture(str(result.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video {result.video}")

    # Sampled frames are sparse relative to source frames; hold the most recent
    # sample's overlay until the next one arrives, so the video plays smoothly.
    sample_of_frame = {
        record.frame_index: index for index, record in enumerate(result.frames)
    }
    ordered_frames = sorted(sample_of_frame)

    written = 0
    try:
        frame_index, current = 0, -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            while (
                current + 1 < len(ordered_frames)
                and ordered_frames[current + 1] <= frame_index
            ):
                current += 1

            canvas = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            # The stable core, tinted, so it is obvious what the box encloses.
            tint = cv2.resize(
                core.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_NEAREST
            )
            overlay = canvas.copy()
            overlay[tint > 0] = _CORE_COLOR
            canvas = cv2.addWeighted(overlay, _CORE_ALPHA, canvas, 1 - _CORE_ALPHA, 0)

            _draw_box(canvas, static.box, scale, _STATIC_COLOR, 2)
            _emphasise_edge(canvas, static.box, scale, _STATIC_COLOR)

            sample_index = (
                sample_of_frame.get(ordered_frames[current]) if current >= 0 else None
            )
            live = rolling[sample_index] if sample_index is not None else None
            if live is not None:
                _draw_box(canvas, live, scale, _ROLLING_COLOR, 1)

            canvas = _with_panel(canvas, panel, static, live, report, window_seconds)
            writer.write(canvas)
            written += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    log.info("wrote %s (%d frames)", out_path, written)
    report["static_box"] = static.box
    report["static_centroid"] = static.centroid
    report["occupancy_cutoff"] = static.threshold
    report["core_pixels"] = static.core_pixels
    report["core_components"] = static.components
    report["output"] = str(out_path)
    report["frames_written"] = written
    return report


def _draw_box(canvas, box: Box, scale: float, colour, thickness: int) -> None:
    x0, y0, x1, y1 = (round(v * scale) for v in box)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)


def _emphasise_edge(canvas, box: Box, scale: float, colour) -> None:
    """Thicken the right and bottom edges: the ones cues should be written against."""
    x0, y0, x1, y1 = (round(v * scale) for v in box)
    cv2.line(canvas, (x1, y0), (x1, y1), colour, 4)
    cv2.line(canvas, (x0, y1), (x1, y1), colour, 4)


def _with_panel(canvas, panel_height: int, static: CabinBox, live, report, window_seconds):
    width = canvas.shape[1]
    panel = np.full((panel_height, width, 3), _PANEL, dtype=np.uint8)
    rolling_std = report.get("rolling_edge_std", {})

    lines = [
        f"static cabin box  x0={static.box[0]:.0f} y0={static.box[1]:.0f} "
        f"x1={static.box[2]:.0f} y1={static.box[3]:.0f}   "
        f"core {static.core_pixels} px, {static.components} component(s), "
        f"occupancy cutoff {static.threshold:.2f}",
        (
            f"rolling box ({window_seconds:.0f}s window)  "
            + (
                f"x0={live[0]:.0f} y0={live[1]:.0f} x1={live[2]:.0f} y1={live[3]:.0f}"
                if live is not None
                else "(window runs off the clip)"
            )
        ),
        "rolling edge std   "
        + "  ".join(f"{name} {rolling_std.get(name, float('nan')):.1f}px" for name in
                    ("left", "top", "right", "bottom"))
        + "    <- right/bottom are the edges FSM cues should use",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (10, 20 + row * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            _TEXT,
            1,
            cv2.LINE_AA,
        )
    return np.vstack([canvas, panel])
