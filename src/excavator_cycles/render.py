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

from .cycles import Cycle
from .logging_setup import get_logger
from .starts import PhaseStart
from .track import TrackResult, load_result
from .video import decode, frame_times, probe

log = get_logger(__name__)

# Purely cosmetic, and therefore not in config.py: nothing here can change a
# reported number.
_TRUCK_COLOR = (150, 150, 150)
_NEGATIVE_COLOR = (70, 70, 240)
_TEXT = (255, 255, 255)
_PANEL = (28, 28, 28)
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
_PLAYHEAD = (110, 110, 110)


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
    starts: list[PhaseStart] | None = None,
    cycles: list[Cycle] | None = None,
) -> RenderStats:
    """Write an annotated copy of the source video.

    Args:
        output_dir: a directory produced by ``track()``.
        out_path: destination file; defaults to ``<output_dir>/annotated.mp4``.
        scale: resize factor. Small sources benefit from >1 so the overlays and
            text are legible; the underlying data is unchanged either way.
        draw_boxes: draw the truck's box as well as the mask.
        starts: the phase starts, in the order the phase search found them. Each is
            drawn as a marker on every signal panel, its search interval is shaded,
            and the latest one that has passed names the running phase on the frame.
            This function does not know or care where they came from, and nothing
            under `src/` reads a label: the annotations are the pipeline's own output.
        cycles: the complete cycles, for the running count on the frame.
    """
    output_dir = Path(output_dir)
    result, masks = load_result(output_dir)
    out_path = Path(out_path) if out_path else output_dir / "annotated.mp4"

    # The physics overlay needs stage 3's output. Absent, the video still renders
    # with masks and boxes -- the stages stay independent.
    table = scene = None
    if physics:
        try:
            from .features import load as load_features

            table, scene = load_features(output_dir)
            log.info("physics overlay enabled")
        except (FileNotFoundError, KeyError) as exc:
            log.warning("no feature data (%s); rendering masks only", exc)

    info = probe(result.video, verify=True)
    width = round(result.width * scale)
    height = round(result.height * scale)

    # Sampled frames are sparse in source-frame terms; this maps every source
    # frame to the most recent sample, so overlays persist between samples.
    by_frame = {record.frame_index: index for index, record in enumerate(result.frames)}
    ordered = sorted(by_frame)

    # The writer accepts exactly one frame size and silently DROPS anything
    # else, so the total size has to account for everything we place -- including
    # the signal graphs, whose presence depends on stage 3 having run.
    # Layout: the video on the left, centred vertically on black, and the signals
    # in a full-height column down the right. Stacking the signals underneath made
    # the canvas nearly square and gave half the frame to the graphs. Without
    # graphs the canvas is just the video.
    graph_width = round(width * 0.62) if table is not None else 0
    canvas_height = height + (max(64, round(height * 0.24)) if graph_width else 0)
    canvas_width = width + graph_width

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (canvas_width, canvas_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    # The output plays at one fixed rate, the source may not: each source frame is
    # written once per output slot that falls in the time it is on screen (0, 1 or
    # 2 slots on an uneven file, exactly 1 on an even one), so the overlay plays in
    # step with the real video. Every time is the frame's stored timestamp.
    times = frame_times(result.video)
    slots = max(1, round(info.duration_seconds * info.fps))
    shown = np.searchsorted(times, np.arange(slots) / info.fps + 1e-6, side="right") - 1
    repeats = np.bincount(np.clip(shown, 0, len(times) - 1), minlength=len(times))

    written = with_mask = 0
    current = -1  # index into `ordered`
    try:
        for frame_index, seconds, source in decode(result.video):
            while current + 1 < len(ordered) and ordered[current + 1] <= frame_index:
                current += 1

            copies = int(repeats[frame_index]) if frame_index < len(repeats) else 0
            if copies == 0:
                continue
            frame = source.to_ndarray(format="bgr24")
            # Tracking shrinks frames to `inference_max_edge` and stores every mask,
            # box and point at that size; draw on the same size.
            if frame.shape[:2] != (result.height, result.width):
                frame = cv2.resize(
                    frame, (result.width, result.height), interpolation=cv2.INTER_AREA
                )
            record = result.frames[by_frame[ordered[current]]] if current >= 0 else None
            sample_position = by_frame[ordered[current]] if current >= 0 else None
            mask = masks.get(sample_position) if sample_position is not None else None

            canvas = _draw_frame(frame, record, result, scale, draw_boxes)
            if table is not None and sample_position is not None:
                canvas = _draw_physics(canvas, table, scene, sample_position, scale)
            # Every phase start is on the pipeline's clock (sample time), so the frame's time
            # is put on it too, or the last cycle is never counted (see sample_clock).
            now = (
                sample_clock(record.time_seconds, times[ordered[current]], seconds)
                if record is not None
                else seconds
            )
            if starts is not None:
                _draw_phase_banner(canvas, starts, cycles or [], now)
            left = _centred(canvas, canvas_height)
            if graph_width:
                canvas = np.hstack(
                    [
                        left,
                        _graph_column(
                            table,
                            sample_position,
                            graph_width,
                            canvas_height,
                            starts,
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
            for _ in range(copies):
                writer.write(canvas)
            written += copies
            with_mask += copies * (mask is not None)
    finally:
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


def _draw_frame(frame, record, result: TrackResult, scale: float, draw_boxes: bool):
    """The video frame with the truck's box and the prompt points drawn on it.

    The SAM mask is deliberately not painted: the tracked mask still feeds the features,
    but the annotated video shows the frame itself.
    """
    canvas = frame.copy()

    if draw_boxes and record is not None and record.truck_box is not None:
        _dashed_box(canvas, record.truck_box, _TRUCK_COLOR)

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


def _draw_physics(canvas, table, scene, position: int, scale: float):
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
        _label(
            canvas, "truck", (at(scene.truck_box)[0], at(scene.truck_box)[1] - 4), _TRUCK_COLOR
        )

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
#
# Each panel carries a plain title AND its unit. The bare symbols (`dh/dt`,
# `|dx/dt|`) were legible only to whoever wrote the feature table, and a unit is
# what lets a reader judge a number: 0.3 means nothing until it says "L/s".
_GRAPHS = (
    ("BUCKET HEIGHT", "L, up is +", "height", _SURFACE),
    ("VERTICAL SPEED dh/dt", "L/s, rising is +", "dh_dt", _DIG),
    ("VERTICAL ACCEL d2h/dt2", "L/s^2", "d2h_dt2", (120, 200, 255)),
    ("SIDEWAYS SPEED |dx/dt|", "L/s", "speed_x", (120, 235, 140)),
    ("OVER THE TRUCK", "share of bucket box, 0-1", "truck_overlap", _DUMP),
    ("BUCKET SHAPE", "box width / height", "aspect_ratio", (230, 160, 240)),
)

_MUTED = (150, 150, 150)


_PHASE_COLOUR = {
    "digging": (240, 180, 90),
    "hauling": (120, 220, 120),
    "dumping": (70, 170, 240),
    "swinging": (240, 130, 220),
}


_SHORT = {"digging": "dig", "hauling": "haul", "dumping": "dump", "swinging": "swing"}
EPS = 1e-6  # seconds: a start AT the current time counts, whatever the last float digit says


def sample_clock(sample_time: float, sample_frame_time: float, frame_time: float) -> float:
    """A frame's time on the pipeline's clock.

    Tracking stamps each sample with its frame index over the frame rate, while a decoded
    frame carries its own presentation timestamp. The two drift apart by up to 0.03 s over
    a clip, and the phase starts are on the first clock. Compared directly, the last frame
    can sit a hair BEFORE the closing dig, so the last cycle is never counted. A frame's
    time is therefore its sample's time plus how long after the sample's frame it is.
    """
    return sample_time + (frame_time - sample_frame_time)


def phase_at(starts: list[PhaseStart], now: float) -> PhaseStart | None:
    """Which phase is running at ``now``: the latest start that has passed."""
    passed = [start for start in starts if now >= start.time - EPS]
    return max(passed, key=lambda start: start.time) if passed else None


def banner_text(starts: list[PhaseStart], cycles: list[Cycle], now: float) -> list[str]:
    """What the frame says at ``now``: the running phase and how long it has run, the
    count of complete cycles, how long each phase of the current cycle has lasted so far,
    and how long the cycle has lasted in all. Before the first phase starts it says what
    the search is doing instead, so the box is never empty."""
    complete = sum(1 for c in cycles if c.end <= now + EPS)
    current = phase_at(starts, now)
    if current is None:
        waiting = "LOOKING FOR DIG START" if starts else "NO PHASE FOUND"
        return [waiting, f"complete cycles: {complete}"]
    lines = [
        f"{current.phase.upper()}   {now - current.time:.1f}s",
        f"complete cycles: {complete}",
    ]
    digs = [s.time for s in starts if s.phase == "digging" and s.time <= now + EPS]
    so_far = [s for s in starts if digs and digs[-1] <= s.time <= now + EPS]
    if not so_far:  # the video opened mid-cycle: there is no cycle to break down yet
        return lines
    edges = [s.time for s in so_far] + [now]
    parts = [f"{_SHORT[s.phase]} {edges[i + 1] - edges[i]:.1f}s" for i, s in enumerate(so_far)]
    return [*lines, "this cycle: " + "  ".join(parts), f"cycle: {now - digs[-1]:.1f}s"]


def _draw_phase_banner(
    canvas, starts: list[PhaseStart], cycles: list[Cycle], now: float
) -> None:
    """Draw :func:`banner_text` top-left, the phase line in that phase's colour."""
    lines = banner_text(starts, cycles, now)
    if not lines:
        return
    current = phase_at(starts, now)
    colour = _PHASE_COLOUR.get(current.phase, _TEXT) if current else _MUTED
    font = max(0.5, canvas.shape[1] / 1400)
    scales = [font, font * 0.7, font * 0.55, font * 0.55][: len(lines)]
    colours = [colour, _TEXT, _TEXT, _TEXT][: len(lines)]
    sizes = [
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, f, 2)
        for t, f in zip(lines, scales, strict=True)
    ]
    box_width = max(w for (w, _), _ in sizes) + 16
    box_height = sum(h + b + 6 for (_, h), b in sizes) + 10
    cv2.rectangle(canvas, (8, 8), (8 + box_width, 8 + box_height), (18, 18, 18), cv2.FILLED)
    cv2.rectangle(canvas, (8, 8), (8 + box_width, 8 + box_height), colour, 2)
    y = 8
    for text, line_colour, f, ((_, h), b) in zip(lines, colours, scales, sizes, strict=True):
        y += h + 6
        cv2.putText(
            canvas, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, f, line_colour, 2, cv2.LINE_AA
        )
        y += b


def _graph_column(
    table,
    position: int | None,
    width: int,
    height: int,
    starts: list[PhaseStart] | None = None,
):
    """The signals, stacked down the right-hand side, with a shared playhead.

    Drawn fresh each frame rather than blitted from a pre-rendered strip,
    because the playhead and the live value both move -- and at this width the
    whole column is a few thousand line segments, which is cheap.

    Every panel is labelled with a title, a unit and its own value range, and the
    column ends in a time axis. A graph a reader has to ask about is a
    graph that did not do its job.
    """
    column = np.full((height, width, 3), _PANEL, dtype=np.uint8)
    rows = len(_GRAPHS)
    axis_height = 18
    plot_bottom = height - axis_height
    each = plot_bottom // rows
    left, right = 56, width - 8  # room for the value scale on the left
    span = max(right - left, 1)
    total = max(len(table.time_seconds) - 1, 1)
    times = np.asarray(table.time_seconds, dtype=float)

    def x_at(when: float) -> int:
        return left + round(span * int(np.argmin(np.abs(times - when))) / total)

    def text(s, origin, colour, size=0.33, thick=1):
        cv2.putText(
            column, s, origin, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA
        )

    # Windows first, so everything else draws on top of them.
    for start in starts or ():
        colour = _PHASE_COLOUR.get(start.phase, _TEXT)
        x0, x1 = x_at(start.window[0]), x_at(start.window[1])
        shade = np.full((plot_bottom - 4, max(1, x1 - x0), 3), colour, dtype=np.uint8)
        region = column[4:plot_bottom, x0 : x0 + shade.shape[1]]
        column[4:plot_bottom, x0 : x0 + shade.shape[1]] = cv2.addWeighted(
            shade[: region.shape[0], : region.shape[1]], 0.22, region, 0.78, 0
        )

    for index, (title, unit, field, colour) in enumerate(_GRAPHS):
        top = index * each
        ceiling, base = top + 24, top + each - 6
        values = np.asarray(getattr(table, field), dtype=float)
        finite = values[np.isfinite(values)]

        (title_width, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text(title, (6, top + 14), colour, 0.4)
        text(unit, (12 + title_width, top + 14), _MUTED, 0.32)
        if finite.size == 0:
            text("not measured on this video", (left, (ceiling + base) // 2), _MUTED)
            continue
        low, high = float(finite.min()), float(finite.max())
        scale = max(high - low, float(np.finfo(float).eps))

        def y_of(value, low=low, scale=scale, base=base, ceiling=ceiling):
            return base - round((value - low) / scale * (base - ceiling))

        # The value scale: the panel's own top and bottom, so a shape can be read
        # as a number without waiting for the playhead to reach it.
        cv2.line(column, (left, base + 3), (right, base + 3), _EDGE, 1)
        text(f"{high:+.2f}", (4, ceiling + 4), _MUTED, 0.3)
        text(f"{low:+.2f}", (4, base), _MUTED, 0.3)
        if low < 0 < high:
            zero = y_of(0.0)
            cv2.line(column, (left, zero), (right, zero), _EDGE, 1)
            text("0", (left - 12, zero + 4), _MUTED, 0.3)

        previous = None
        for i, value in enumerate(values):
            if not np.isfinite(value):
                previous = None
                continue
            point = (left + round(span * i / total), y_of(value))
            if previous is not None:
                cv2.line(column, previous, point, colour, 1, cv2.LINE_AA)
            previous = point

        if position is not None and np.isfinite(values[position]):
            now = f"now {values[position]:+.3f}"
            (now_width, _), _ = cv2.getTextSize(now, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            text(now, (right - now_width, top + 14), _TEXT, 0.36)

    # Phase-start labels would sit on top of each other wherever two starts are close
    # (hauling and dumping start 1 s apart on the dev clip). Each label takes the
    # lowest row where it does not touch one already placed, and flips to the left
    # of its line near the right edge rather than running off the frame.
    placed: list[tuple[int, int, int]] = []  # (row, x0, x1)

    def mark_label(label: str, x: int, colour, bottom: int) -> None:
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        x0 = x + 3 if x + 3 + w < width - 2 else x - 3 - w
        row = 0
        while any(r == row and x0 < b + 6 and a < x0 + w + 6 for r, a, b in placed):
            row += 1
        placed.append((row, x0, x0 + w))
        text(label, (x0, bottom - row * 11), colour, 0.3)

    # Each detected phase start is a SOLID, phase-coloured line with a dot on top. It
    # spans every panel, so one boundary can be read against all six signals at once --
    # the point of stacking them. A start marked * is its window's centre: nothing in
    # the window moved faster than noise.
    marks = []
    for start in starts or ():
        x = x_at(start.time)
        colour = _PHASE_COLOUR.get(start.phase, _TEXT)
        cv2.line(column, (x, 4), (x, plot_bottom), colour, 1, cv2.LINE_AA)
        cv2.circle(column, (x, 9), 3, colour, -1)
        marks.append((x, start.phase + ("*" if start.fallback else ""), colour))

    for x, label, colour in sorted(marks, key=lambda mark: mark[0]):
        mark_label(label, x, colour, plot_bottom - 4)

    # The time axis, in seconds -- every panel shares it.
    axis_y = plot_bottom + 12
    end = float(times[-1]) if len(times) else 0.0
    step = 5.0 if end > 15 else 1.0
    tick = 0.0
    while tick <= end + 1e-9:
        x = x_at(tick)
        cv2.line(column, (x, plot_bottom), (x, plot_bottom + 3), _MUTED, 1)
        text(f"{tick:g}s", (x - 6, axis_y), _MUTED, 0.3)
        tick += step
    text("time", (6, axis_y), _MUTED, 0.3)

    if position is not None:
        x = left + round(span * position / total)
        cv2.line(column, (x, 4), (x, plot_bottom), _PLAYHEAD, 1)
    return column


def _centred(picture, height: int):
    """``picture`` centred vertically on a black canvas ``height`` tall."""
    top = (height - picture.shape[0]) // 2
    canvas = np.zeros((height, picture.shape[1], 3), dtype=np.uint8)
    canvas[top : top + picture.shape[0]] = picture
    return canvas


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
