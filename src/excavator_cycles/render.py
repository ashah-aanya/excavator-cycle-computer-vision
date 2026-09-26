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
_EDGE = (60, 60, 60)
_PLAYHEAD = (160, 160, 160)


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
    onsets: dict[str, float] | None = None,
) -> RenderStats:
    """Write an annotated copy of the source video.

    Args:
        output_dir: a directory produced by ``track()``.
        out_path: destination file; defaults to ``<output_dir>/annotated.mp4``.
        scale: resize factor. Small sources benefit from >1 so the overlays and
            text are legible; the underlying data is unchanged either way.
        draw_boxes: include the detector's boxes as well as the mask.
        onsets: phase name -> onset time in seconds. Drawn as a marker on every
            signal panel and as a phase banner on the frame. This function does
            not know or care where they came from: the state machine will supply
            predictions, and `eval/annotate_solution.py` supplies the hand labels
            to make a reference video. Nothing under `src/` may read the labels
            itself, so they arrive as an argument or not at all.
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

            table, scene = load_features(output_dir)
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
    # Layout: the video on the left with a thin status bar under it, and the
    # signals in a column down the right. Stacking the signals underneath made
    # the canvas nearly square and gave half the frame to the graphs.
    graph_width = round(width * 0.62) if strip is not None else 0
    canvas_height = height + panel_height
    canvas_width = width + graph_width

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (canvas_width, canvas_height),
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
            if onsets:
                _draw_phase_banner(canvas, onsets, frame_index / info.fps)
            left = np.vstack(
                [
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
            )
            if graph_width:
                canvas = np.hstack(
                    [
                        left,
                        _graph_column(
                            table, sample_position, graph_width, canvas_height, onsets
                        ),
                    ]
                )
            else:
                canvas = left
            if canvas.shape[:2] != (canvas_height, canvas_width):
                # Without this the writer drops the frame and returns nothing,
                # and the run reports success while producing an empty file.
                raise RuntimeError(
                    f"frame is {canvas.shape[1]}x{canvas.shape[0]}, "
                    f"writer expects {canvas_width}x{canvas_height}"
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
    """Draw what the measurement stage actually used: two boxes and a pivot.

    Every shape here is read from the feature table, so the video cannot show
    something the pipeline did not measure. That is the point -- the task
    requires the annotation to be evidence, not illustration.
    """

    def at(box):
        return tuple(round(v * scale) for v in box)

    pivot = (int(scene.pivot[0] * scale), int(scene.pivot[1] * scale))

    if scene.truck_box is not None:
        _dashed_box(canvas, at(scene.truck_box), _TRUCK_COLOR)
        _label(canvas, "truck", (at(scene.truck_box)[0], at(scene.truck_box)[1] - 4),
               _TRUCK_COLOR)

    cabin = table.cabin_box[position]
    if np.isfinite(cabin).all():
        _box(canvas, at(cabin), _FOREARM_BAND, 1)
        _label(canvas, "cabin", (at(cabin)[0], at(cabin)[1] - 4), _FOREARM_BAND)

    bucket = table.bucket_box[position]
    if np.isfinite(bucket).all():
        _box(canvas, at(bucket), _BUCKET_BAND, 2)
        centre = (
            round((bucket[0] + bucket[2]) / 2 * scale),
            round((bucket[1] + bucket[3]) / 2 * scale),
        )
        cv2.circle(canvas, centre, 3, _TIP, -1)
        cv2.line(canvas, pivot, centre, _TRACE, 1)

        # Where the bucket centre has been, so a jump is visible rather than
        # merely implied by a number changing.
        trail = table.bucket_box[max(0, position - 25) : position + 1]
        points = [
            (round((b[0] + b[2]) / 2 * scale), round((b[1] + b[3]) / 2 * scale))
            for b in trail
            if np.isfinite(b).all()
        ]
        for start, end in pairwise(points):
            cv2.line(canvas, start, end, _TRACE, 1)

    cv2.drawMarker(canvas, pivot, _PIVOT, cv2.MARKER_CROSS, 12, 2)
    _label(canvas, "slew centre", (pivot[0] + 8, pivot[1] - 6), _PIVOT)
    return canvas


# Which signals ride along with the video, and in what order. Six rather than
# all thirteen: these are the ones a phase boundary will be read off, and a
# panel too short to see a shape in is worse than no panel.
_GRAPHS = (
    ("height  (up +)", "height", _SURFACE),
    ("dh/dt", "dh_dt", _DIG),
    ("d2h/dt2", "d2h_dt2", (120, 200, 255)),
    ("|dx/dt|", "speed_x", (120, 235, 140)),
    ("truck overlap", "truck_overlap", _DUMP),
    ("aspect ratio", "aspect_ratio", (230, 160, 240)),
)


def _signal_strip(table, scene, height: int = 0):
    """Kept so `render` can test whether stage 3 ran; the drawing is per-frame."""
    return table


_PHASE_COLOUR = {
    "digging": (240, 180, 90),
    "hauling": (120, 220, 120),
    "dumping": (70, 170, 240),
    "swinging": (240, 130, 220),
}


def phase_at(onsets: dict[str, float], now: float) -> str | None:
    """Which phase is running at ``now``: the latest onset that has passed."""
    passed = [
        (when, name) for name, when in onsets.items() if when is not None and now >= when
    ]
    return max(passed)[1] if passed else None


def _draw_phase_banner(canvas, onsets: dict[str, float], now: float) -> None:
    """The current phase, and how long it has been running."""
    name = phase_at(onsets, now)
    if name is None:
        return
    started = onsets[name]
    colour = _PHASE_COLOUR.get(name, _TEXT)
    text = f"{name.upper()}   {now - started:.1f}s"
    font = max(0.5, canvas.shape[1] / 1400)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 2)
    cv2.rectangle(canvas, (8, 8), (8 + tw + 16, 8 + th + base + 12), (18, 18, 18), cv2.FILLED)
    cv2.rectangle(canvas, (8, 8), (8 + tw + 16, 8 + th + base + 12), colour, 2)
    cv2.putText(canvas, text, (16, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, font, colour, 2,
                cv2.LINE_AA)


def _graph_column(
    table,
    position: int | None,
    width: int,
    height: int,
    onsets: dict[str, float] | None = None,
):
    """The signals, stacked down the right-hand side, with a shared playhead.

    Drawn fresh each frame rather than blitted from a pre-rendered strip,
    because the playhead and the live value both move -- and at this width the
    whole column is a few thousand line segments, which is cheap.
    """
    column = np.full((height, width, 3), _PANEL, dtype=np.uint8)
    rows = len(_GRAPHS)
    each = height // rows
    left, right = 56, width - 8  # room for the label on the left
    span = max(right - left, 1)
    total = max(len(table.time_seconds) - 1, 1)

    for index, (label, field, colour) in enumerate(_GRAPHS):
        top = index * each
        base, ceiling = top + each - 10, top + 8
        values = np.asarray(getattr(table, field), dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = float(finite.min()), float(finite.max())
        scale = max(high - low, float(np.finfo(float).eps))

        cv2.line(column, (left, base + 4), (right, base + 4), _EDGE, 1)
        previous = None
        for i, value in enumerate(values):
            if not np.isfinite(value):
                previous = None
                continue
            x = left + round(span * i / total)
            y = base - round((value - low) / scale * (base - ceiling))
            if previous is not None:
                cv2.line(column, previous, (x, y), colour, 1, cv2.LINE_AA)
            previous = (x, y)

        cv2.putText(column, label, (6, top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, colour, 1, cv2.LINE_AA)
        if position is not None and np.isfinite(values[position]):
            cv2.putText(column, f"{values[position]:+.3f}", (6, top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, _TEXT, 1, cv2.LINE_AA)

    # Onset markers span every panel, so a boundary can be read against all six
    # signals at once -- which is the point of stacking them.
    times = np.asarray(table.time_seconds, dtype=float)
    for name, when in (onsets or {}).items():
        if when is None:
            continue
        index = int(np.argmin(np.abs(times - when)))
        x = left + round(span * index / total)
        colour = _PHASE_COLOUR.get(name, _TEXT)
        for y in range(4, height - 4, 6):  # dashed, so it reads under the traces
            cv2.line(column, (x, y), (x, min(y + 3, height - 4)), colour, 1)
        cv2.putText(column, name[:4], (x + 3, height - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, colour, 1, cv2.LINE_AA)

    if position is not None:
        x = left + round(span * position / total)
        cv2.line(column, (x, 4), (x, height - 4), _PLAYHEAD, 1)
    return column


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
    # Sized against the video's width, not the panel's height: the panel is a
    # thin status bar now that the signals live in their own column.
    scale = max(0.42, width / 1500)
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
        scale * 0.62,
        colour,
        1,
        cv2.LINE_AA,
    )

    if table is not None and position is not None:
        physics = (
            f"h {float(table.height[position]):+.3f} L   "
            f"|dx/dt| {float(table.speed_x[position]):.3f} L/s   "
            f"overlap {float(table.truck_overlap[position]):.2f}   "
            f"AR {float(table.aspect_ratio[position]):.2f}"
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
    # Against the canvas WIDTH: the overlay labels sit on the video, which is
    # wide and short, and sizing them off its height made them shout.
    font_scale = max(0.28, canvas.shape[1] / 2600)
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
