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

    info = probe(result.video, verify=True)
    width = round(result.width * scale)
    height = round(result.height * scale)
    panel_height = max(46, round(height * 0.16))

    # Sampled frames are sparse in source-frame terms; this maps every source
    # frame to the most recent sample, so overlays persist between samples.
    by_frame = {record.frame_index: index for index, record in enumerate(result.frames)}
    ordered = sorted(by_frame)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (width, height + panel_height),
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
            canvas = np.vstack(
                [
                    canvas,
                    _draw_panel(record, result, width, panel_height, frame_index, info.fps),
                ]
            )
            writer.write(canvas)
            written += 1
            with_mask += mask is not None
            frame_index += 1
    finally:
        capture.release()
        writer.release()

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


def _draw_panel(
    record, result: TrackResult, width: int, height: int, frame_index: int, fps: float
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
