"""Rebuild the video with the pipeline's own decisions drawn on it.

This reads only the cache written by stage 2. It never loads a model, which is
what lets the expensive work happen once on a GPU node while the annotation is
iterated on a laptop in seconds.

It is also the deliverable's evidence. The task requires the annotations to be
the pipeline's real output rather than anything corrected afterwards, so every
overlay here is drawn from the stored mask and the stored detections -- if the
mask is wrong, the video shows it wrong.

Overlays are held between samples rather than interpolated. Tracking runs at
10 Hz and the source is 30 Hz, so each mask covers about three frames. Inventing
intermediate shapes would draw something the pipeline never computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from .logging_setup import get_logger
from .track import TrackResult, load_result
from .video import probe

log = get_logger(__name__)

# Purely cosmetic, and therefore not in config.py: nothing here can change a
# reported number.
_MASK_COLOR = (80, 230, 90)
_BOX_COLOR = (90, 220, 250)
_TRUCK_COLOR = (150, 150, 150)
_NEGATIVE_COLOR = (70, 70, 240)
_SEED_COLOR = (250, 200, 90)
_TEXT = (255, 255, 255)
_PANEL = (28, 28, 28)
_WARN = (60, 80, 240)
_MASK_ALPHA = 0.45
# The physics overlay: what the geometry stage actually measures.
_PIVOT = (80, 220, 250)
_TIP = (60, 60, 250)
_BUCKET_BAND = (90, 90, 250)
_FOREARM_BAND = (250, 200, 90)
_SURFACE = (60, 140, 255)
_DIG = (90, 230, 120)
_DUMP = (250, 180, 80)
_TRACE = (220, 220, 220)


@dataclass
class RenderStats:
    frames_written: int
    frames_with_mask: int
    output_path: Path


def render(
    output_dir: str | Path,
    out_path: str | Path | None = None,
    scale: float = 1.0,
    draw_boxes: bool = True,
    physics: bool = True,
) -> RenderStats:
    """Write an annotated copy of the source video.

    Args:
        output_dir: a directory produced by ``track()``.
        out_path: destination file; defaults to ``<output_dir>/annotated.mp4``.
        scale: resize factor. Small sources benefit from >1 so the overlays and
            text are legible; the underlying data is unchanged either way.
        draw_boxes: include the detector's boxes as well as the mask.
    """
    output_dir = Path(output_dir)
    result, masks = load_result(output_dir)
    out_path = Path(out_path) if out_path else output_dir / "annotated.mp4"

    # The physics overlay needs stage 3's output. Absent, the video still renders
    # with masks and boxes -- the stages stay independent.
    table = scene = strip = None
    if physics:
        try:
            from .features import load as load_features

            table, scene_dict = load_features(output_dir)
            scene = scene_dict
            strip = _signal_strip(table, scene)
            log.info("physics overlay enabled")
        except (FileNotFoundError, KeyError) as exc:
            log.warning("no feature data (%s); rendering masks only", exc)

    info = probe(result.video, verify=True)
    width = round(result.width * scale)
    height = round(result.height * scale)
    panel_height = max(58, round(height * 0.22))

    # Sampled frames are sparse in source-frame terms; this maps every source
    # frame to the most recent sample, so overlays persist between samples.
    by_frame = {record.frame_index: index for index, record in enumerate(result.frames)}
    ordered = sorted(by_frame)

    # The writer accepts exactly one frame size and silently DROPS anything
    # else, so the total height has to account for every panel we stack --
    # including the signal strip, whose presence depends on stage 3 having run.
    strip_height = strip.shape[0] if strip is not None else 0
    canvas_height = height + panel_height + strip_height

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (width, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    capture = cv2.VideoCapture(str(result.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video {result.video}")

    written = with_mask = 0
    current = -1  # index into `ordered`
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            while current + 1 < len(ordered) and ordered[current + 1] <= frame_index:
                current += 1

            record = result.frames[by_frame[ordered[current]]] if current >= 0 else None
            sample_position = by_frame[ordered[current]] if current >= 0 else None
            mask = masks.get(sample_position) if sample_position is not None else None

            canvas = _draw_frame(frame, mask, record, result, scale, draw_boxes)
            if table is not None and sample_position is not None:
                canvas = _draw_physics(canvas, mask, table, scene, sample_position, scale)
            panels = [
                canvas,
                _draw_panel(
                    record,
                    result,
                    width,
                    panel_height,
                    frame_index,
                    info.fps,
                    table,
                    sample_position,
                ),
            ]
            if strip is not None:
                panels.append(_blit_strip(strip, table, sample_position, width))
            canvas = np.vstack(panels)
            if canvas.shape[:2] != (canvas_height, width):
                # Without this the writer drops the frame and returns nothing,
                # and the run reports success while producing an empty file.
                raise RuntimeError(
                    f"frame is {canvas.shape[1]}x{canvas.shape[0]}, "
                    f"writer expects {width}x{canvas_height}"
                )
            writer.write(canvas)
            written += 1
            with_mask += mask is not None
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    size = out_path.stat().st_size if out_path.exists() else 0
    if size < 10_000:
        raise RuntimeError(
            f"{out_path} is {size} bytes after writing {written} frames; "
            "the encoder rejected them"
        )
    log.info("wrote %s (%d frames, %d with a mask)", out_path, written, with_mask)
    return RenderStats(
        frames_written=written, frames_with_mask=with_mask, output_path=out_path
    )


def _draw_frame(frame, mask, record, result: TrackResult, scale: float, draw_boxes: bool):
    """The video frame with mask, boxes and prompt points drawn on it."""
    canvas = frame.copy()

    if mask is not None and mask.shape[:2] == canvas.shape[:2]:
        tint = np.array(_MASK_COLOR, dtype=np.float32)
        canvas[mask] = ((1 - _MASK_ALPHA) * canvas[mask] + _MASK_ALPHA * tint).astype(np.uint8)
        # Outline makes the mask boundary legible where the tint alone is subtle.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, _MASK_COLOR, 1)

    if draw_boxes and record is not None:
        if record.truck_box is not None:
            _dashed_box(canvas, record.truck_box, _TRUCK_COLOR)
        if record.detection_box is not None:
            merged = (record.detection_truck_iou or 0.0) >= 0.3
            colour = _WARN if merged else _BOX_COLOR
            _box(canvas, record.detection_box, colour)
            label = f"det {record.detection_score:.2f}" + (" MERGED" if merged else "")
            _label(canvas, label, record.detection_box[:2], colour)

    # The prompt that produced every mask in this video: worth seeing, because
    # a bad seed explains everything downstream.
    for point in result.seed["negative_points"]:
        cv2.drawMarker(
            canvas,
            (int(point[0]), int(point[1])),
            _NEGATIVE_COLOR,
            cv2.MARKER_TILTED_CROSS,
            7,
            1,
        )

    if scale != 1.0:
        canvas = cv2.resize(
            canvas,
            (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)),
            interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA,
        )
    return canvas


def _draw_physics(canvas, mask, table, scene, position: int, scale: float):
    """Draw what the geometry stage measures, on top of the frame.

    The point is that the derived quantities are checkable: the pivot should sit
    on the machine's body, the arm line should follow the boom, the outer band
    should cover the bucket and the inner one the stick, and the surface line
    should lie on the material.
    """
    import numpy as np

    from .geometry import radial_band

    centre = (scene["centre"][0] * scale, scene["centre"][1] * scale)
    reach = scene["scale"] * scale

    if mask is not None:
        # Tint the two bands the curl is measured from, so the reader can see
        # exactly which pixels produced the angle.
        for band, colour in (
            ((scene_band_low(scene), 1.0), _BUCKET_BAND),
            ((scene_forearm_low(scene), scene_band_low(scene)), _FOREARM_BAND),
        ):
            region = radial_band(mask, tuple(scene["centre"]), *band)
            if not region.any():
                continue
            big = cv2.resize(
                region.astype(np.uint8),
                (canvas.shape[1], canvas.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            canvas[big] = (0.5 * np.array(colour) + 0.5 * canvas[big]).astype(np.uint8)

    # Pivot, scale circle, and the arm line out to the measured bucket point.
    cv2.circle(canvas, (int(centre[0]), int(centre[1])), int(reach), _PIVOT, 1, cv2.LINE_AA)
    cv2.drawMarker(
        canvas, (int(centre[0]), int(centre[1])), _PIVOT, cv2.MARKER_CROSS, int(14 * scale), 2
    )

    bearing = float(table.bearing[position])
    extension = float(table.extension[position]) * reach
    tip = (
        int(centre[0] + np.cos(bearing) * extension),
        int(centre[1] - np.sin(bearing) * extension),
    )
    cv2.line(canvas, (int(centre[0]), int(centre[1])), tip, _TRACE, 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, tip, _TIP, cv2.MARKER_TILTED_CROSS, int(12 * scale), 2)

    # The material surface: the level that defines two of the four boundaries.
    if scene.get("surface_height") is not None:
        y = int(centre[1] - scene["surface_height"] * reach)
        cv2.line(canvas, (0, y), (canvas.shape[1], y), _SURFACE, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "material surface",
            (6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32 * scale,
            _SURFACE,
            1,
            cv2.LINE_AA,
        )

    for key, colour, label in (("dig_zone", _DIG, "dig"), ("dump_zone", _DUMP, "dump")):
        zone = scene.get(key)
        if zone:
            point = (int(zone[0] * scale), int(zone[1] * scale))
            cv2.circle(canvas, point, int(6 * scale), colour, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (point[0] + 8, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32 * scale,
                colour,
                1,
                cv2.LINE_AA,
            )
    return canvas


def scene_band_low(scene) -> float:
    return float(scene.get("bucket_band_low", 0.85))


def scene_forearm_low(scene) -> float:
    return float(scene.get("forearm_band_low", 0.55))


def _signal_strip(table, scene, height: int = 96):
    """Pre-render the whole video's signals once; the renderer blits a window."""
    import numpy as np

    n = len(table.time_seconds)
    strip = np.full((height, n, 3), 22, dtype=np.uint8)
    rows = [
        (
            "elevation",
            np.asarray(table.elevation, float),
            _SURFACE,
            None if scene.get("surface_height") is None else float(scene["surface_height"]),
        ),
        ("curl", np.asarray(table.curl, float), _BUCKET_BAND, None),
        ("slew", np.asarray(table.slew_rate, float), _DIG, 0.0),
    ]
    band = height // len(rows)
    for index, (name, values, colour, reference) in enumerate(rows):
        top = index * band
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        lo, hi = float(finite.min()), float(finite.max())
        span = (hi - lo) or 1.0

        def to_y(v, top=top, lo=lo, span=span, band=band):
            return int(top + band - 4 - (v - lo) / span * (band - 8))

        if reference is not None and lo <= reference <= hi:
            y = to_y(reference)
            cv2.line(strip, (0, y), (n, y), (70, 70, 70), 1)
        points = [(x, to_y(v)) for x, v in enumerate(values) if np.isfinite(v)]
        for (x0, y0), (x1, y1) in pairwise(points):
            if x1 - x0 <= 2:
                cv2.line(strip, (x0, y0), (x1, y1), colour, 1, cv2.LINE_AA)
        cv2.putText(
            strip,
            name,
            (3, top + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return strip


def _blit_strip(strip, table, position: int | None, width: int):
    """The signal strip scaled to the frame width, with a playhead."""

    out = cv2.resize(strip, (width, strip.shape[0]), interpolation=cv2.INTER_AREA)
    if position is not None and len(table.time_seconds) > 1:
        x = int(position / (len(table.time_seconds) - 1) * (width - 1))
        cv2.line(out, (x, 0), (x, out.shape[0]), (255, 255, 255), 1)
    return out


def _draw_panel(
    record,
    result: TrackResult,
    width: int,
    height: int,
    frame_index: int,
    fps: float,
    table=None,
    position: int | None = None,
):
    """A readout strip: time, and the numbers QA judges the masks by."""
    panel = np.full((height, width, 3), _PANEL, dtype=np.uint8)
    scale = max(0.35, height / 130)
    line = int(height * 0.42)

    seconds = frame_index / fps if fps else 0.0
    left = [
        f"t {seconds:6.2f}s   frame {frame_index}",
        f"mask {record.mask_area_fraction * 100:5.2f}%   conf {record.sam_confidence:.2f}"
        if record is not None
        else "no sample yet",
    ]
    for index, text in enumerate(left):
        cv2.putText(
            panel,
            text,
            (8, line + index * int(height * 0.38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _TEXT,
            1,
            cv2.LINE_AA,
        )

    status = result.qa["status"].upper()
    colour = _WARN if status == "FAIL" else _MASK_COLOR
    right = f"QA {status}"
    (text_width, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(
        panel,
        right,
        (width - text_width - 10, line),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )

    if table is not None and position is not None:
        physics = (
            f"elev {float(table.elevation[position]):+.3f} L   "
            f"curl {float(table.curl[position]):+.2f} rad   "
            f"slew {float(table.slew_rate[position]):+.2f} rad/s"
        )
        cv2.putText(
            panel,
            physics,
            (8, line + int(height * 0.76)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.85,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )

    seed_text = f"seed t={result.seed['time_seconds']:.1f}s"
    (seed_width, _), _ = cv2.getTextSize(seed_text, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, 1)
    cv2.putText(
        panel,
        seed_text,
        (width - seed_width - 10, line + int(height * 0.38)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.9,
        _SEED_COLOR,
        1,
        cv2.LINE_AA,
    )

    _progress_bar(panel, frame_index, result, width, height, fps)
    return panel


def _progress_bar(panel, frame_index, result: TrackResult, width, height, fps):
    """A thin ribbon: the whole video's timeline with a playhead."""
    y = height - 6
    cv2.line(panel, (8, y), (width - 8, y), (70, 70, 70), 2)
    if result.duration_seconds <= 0:
        return
    position = (frame_index / fps) / result.duration_seconds if fps else 0.0
    x = int(8 + position * (width - 16))
    cv2.line(panel, (x, y - 4), (x, y + 4), _TEXT, 1)

    seed_x = int(8 + (result.seed["time_seconds"] / result.duration_seconds) * (width - 16))
    cv2.drawMarker(panel, (seed_x, y), _SEED_COLOR, cv2.MARKER_TRIANGLE_UP, 7, 1)


def _box(canvas, box, colour, thickness: int = 1):
    x1, y1, x2, y2 = (round(v) for v in box)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)


def _dashed_box(canvas, box, colour, dash: int = 6):
    """Dashed, to signal "supporting evidence only" rather than a tracked object."""
    x1, y1, x2, y2 = (round(v) for v in box)
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), colour, 1)
        cv2.line(canvas, (x, y2), (min(x + dash, x2), y2), colour, 1)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), colour, 1)
        cv2.line(canvas, (x2, y), (x2, min(y + dash, y2)), colour, 1)


def _label(canvas, text, origin, colour):
    font_scale = max(0.3, canvas.shape[0] / 900)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    x, y = int(origin[0]), int(origin[1])
    top = max(0, y - text_height - baseline - 2)
    cv2.rectangle(
        canvas,
        (x, top),
        (x + text_width + 4, top + text_height + baseline + 2),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        canvas,
        text,
        (x + 2, top + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
