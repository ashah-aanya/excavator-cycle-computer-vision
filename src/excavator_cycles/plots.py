"""Diagnostic plots -- the gate for the feature stage.

The rule for this stage is that a human must be able to *see* the four phase
boundaries in these signals before any rule is asked to find them. If a boundary
is invisible here, no threshold will locate it reliably, and the fix belongs in
the features rather than in cleverer logic downstream.

Two figures:

* **signals** -- every measurement against time, so the cycle's rhythm and the
  boundaries are visible directly.
* **scene** -- the derived landmarks drawn on a real frame, so the pivot, the
  scale, the two zones and the surface height can be checked against what a
  person would point at.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display on a cluster node
import matplotlib.pyplot as plt

from .features import FeatureTable
from .geometry import Scene
from .logging_setup import get_logger

log = get_logger(__name__)


def plot_signals(table: FeatureTable, scene: Scene, path: str | Path) -> Path:
    """Every feature against time, stacked on a shared axis."""
    path = Path(path)
    panels = [
        ("bearing (rad)", table.bearing, None),
        ("slew rate (rad/s)\nsign = swing direction", table.slew_rate, 0.0),
        ("elevation (L)\nvs material surface", table.elevation, scene.surface_height),
        ("elevation rate (L/s)", table.elevation_rate, 0.0),
        (
            "curl vs forearm (rad)\ngaps = not measurable",
            _gapped(table.curl, table.curl_valid),
            None,
        ),
        ("curl rate (rad/s)", _gapped(table.curl_rate, table.curl_valid), 0.0),
        ("falling material (flow)\nINDEPENDENT of the mask", table.falling, None),
        ("speed (L/s)", table.speed, None),
        ("mask area (L^2)\ndips when buried", table.area, None),
    ]

    figure, axes = plt.subplots(len(panels), 1, figsize=(14, 2.0 * len(panels)), sharex=True)
    for axis, (label, values, reference) in zip(axes, panels, strict=True):
        axis.plot(table.time_seconds, values, linewidth=1.2, color="#1f4e79")
        if reference is not None:
            axis.axhline(reference, color="#c0392b", linestyle="--", linewidth=0.9)
        axis.set_ylabel(label, fontsize=8)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)

        # Shade where the bucket is in each zone: the cycle's rhythm should be
        # obvious, and if it is not, the zones are wrong.
        _shade(axis, table.time_seconds, table.in_dig_zone, "#2e7d32", "dig")
        _shade(axis, table.time_seconds, table.in_dump_zone, "#1565c0", "dump")

    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    figure.suptitle(
        "Feature signals -- the four phase boundaries should be visible by eye",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def _gapped(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blank out samples that were interpolated rather than measured.

    A filled gap drawn as a solid line reads as data. The curl signal looked
    smooth and trustworthy in an earlier plot precisely because three quarters
    of it was interpolation.
    """
    out = np.asarray(values, dtype=float).copy()
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def _shade(axis, times, flags, colour, label):
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return
    edges = np.diff(flags.astype(int))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if flags[0]:
        starts.insert(0, 0)
    if flags[-1]:
        ends.append(len(flags) - 1)
    for index, (start, end) in enumerate(zip(starts, ends, strict=False)):
        axis.axvspan(
            times[start],
            times[end],
            color=colour,
            alpha=0.10,
            label=label if index == 0 else None,
        )


def plot_scene(frame: np.ndarray, table: FeatureTable, scene: Scene, path: str | Path) -> Path:
    """The derived landmarks drawn on a real frame.

    This is the check that matters most in this stage: the pivot should sit on
    the machine's body, the scale circle should reach about as far as the arm
    does, the two zones should land where the bucket actually works, and the
    surface line should lie on the pile.
    """
    path = Path(path)
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cx, cy = scene.centre
    axis.plot(
        cx,
        cy,
        "+",
        color="#ffeb3b",
        markersize=16,
        markeredgewidth=2.5,
        label="rotation centre",
    )
    axis.add_patch(
        plt.Circle(
            (cx, cy),
            scene.scale,
            fill=False,
            color="#ffeb3b",
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
        )
    )

    for zone, colour, name in (
        (scene.dig_zone, "#4caf50", "dig zone"),
        (scene.dump_zone, "#2196f3", "dump zone"),
    ):
        if zone is not None:
            axis.plot(
                *zone, "o", color=colour, markersize=13, markeredgecolor="white", label=name
            )

    if scene.surface_height is not None and scene.dig_zone is not None:
        y = cy - scene.surface_height * scene.scale
        axis.axhline(
            y, color="#ff5722", linewidth=1.6, linestyle="--", label="material surface"
        )

    # The bucket's whole path, so the cycle is visible as a shape.
    bx = cx + np.cos(table.bearing) * table.extension * scene.scale
    by = cy - np.sin(table.bearing) * table.extension * scene.scale
    axis.plot(bx, by, "-", color="#ffffff", linewidth=0.7, alpha=0.6, label="bucket path")

    axis.set_title(
        f"Derived scene   L = {scene.scale:.0f} px"
        + (f"   zones {scene.zone_separation:.2f} L apart" if scene.zone_separation else "")
        + f"   return = {'left' if scene.return_sign > 0 else 'right'}",
        fontsize=10,
    )
    axis.legend(loc="lower right", fontsize=8, framealpha=0.85)
    axis.set_xlim(0, frame.shape[1])
    axis.set_ylim(frame.shape[0], 0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def plot_motion(field, times: np.ndarray, path: str | Path) -> Path:
    """The motion field as four stacked signals, with the dwell gate shaded.

    Laid out so the three questions the state machine asks can be read off one
    figure: which way is it swinging, is the bucket rising or falling, and is it
    moving at all.
    """
    path = Path(path)
    figure, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    still = np.isfinite(field.speed) & (field.speed < np.nanquantile(field.speed, 0.35))
    panels = (
        (
            field.omega,
            "tab:blue",
            "swing rate (rad/s)",
            "sign separates hauling from returning",
        ),
        (field.v_up, "tab:green", "bucket vertical (L/s)", "positive while the bucket rises"),
        (field.v_radial, "tab:orange", "radial rate (L/s)", "reaching out vs folding in"),
        (field.speed, "tab:purple", "speed (L/s)", "the dwell gate"),
    )
    for axis, (signal, colour, label, note) in zip(axes, panels, strict=True):
        axis.plot(times, signal, lw=0.8, alpha=0.4, color=colour)
        axis.plot(times, _smooth_for_display(signal), lw=2.0, color=colour)
        axis.axhline(0, color="k", lw=0.8)
        _shade(axis, times, still, "0.85", "dwell")
        axis.set_ylabel(label, fontsize=9)
        axis.set_title(note, fontsize=9, loc="left")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle("Motion field: measured from flow, not from arm pose", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def _smooth_for_display(signal: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """Savitzky-Golay over the finite samples, for the eye only.

    Symmetric, so it does not shift an extremum -- but nothing downstream reads
    this; the state machine smooths for itself with the configured window.
    """
    from scipy.signal import savgol_filter

    out = np.asarray(signal, dtype=float).copy()
    good = np.isfinite(out)
    if good.sum() < window:
        return out
    out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return savgol_filter(out, window, order)


def render_flow_overlay(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    field,
    times: np.ndarray,
    pivot: tuple[float, float],
    output_dir: str | Path,
    upscale: int = 3,
    arrow_gain: float = 8.0,
) -> Path:
    """Draw the flow field on the frames, with the measured numbers and a strip.

    This exists because every wrong call on this project was caught by looking
    at a picture. The arrows are the raw evidence; the HUD is what the pipeline
    made of it; disagreeing with each other is the thing worth seeing.
    """
    from .motion import dense_flow

    output_dir = Path(output_dir)
    path = output_dir / "motion.mp4"
    height, width = masks[0].shape
    frame_width, frame_height = width * upscale, height * upscale
    strip_height = 150

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, 1.0 / max(np.median(np.diff(times)), 1e-6)),
        (frame_width, frame_height + strip_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open a writer for {path}")

    curves = [
        (_normalise(_smooth_for_display(field.omega)), (255, 160, 60), 25, "swing"),
        (_normalise(_smooth_for_display(field.v_up)), (60, 220, 60), 70, "bucket up/down"),
        (_normalise(_smooth_for_display(field.speed)), (200, 120, 255), 115, "speed"),
    ]

    written = 0
    for index in range(len(frames) - 1):
        image = cv2.resize(
            frames[index], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST
        )
        mask = masks[index]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, [c * upscale for c in contours], -1, (0, 255, 255), 1)
        cv2.circle(
            image, (int(pivot[0] * upscale), int(pivot[1] * upscale)), 5, (255, 0, 255), -1
        )

        flow = dense_flow(frames[index], frames[index + 1])
        for y in range(0, height, 6):
            for x in range(0, width, 6):
                if not mask[y, x]:
                    continue
                fx, fy = flow[y, x]
                magnitude = float(np.hypot(fx, fy))
                if magnitude < 0.25:
                    continue
                colour = (0, 255, 0) if magnitude > 0.8 else (0, 200, 255)
                cv2.arrowedLine(
                    image,
                    (x * upscale, y * upscale),
                    (
                        int((x + fx * arrow_gain) * upscale),
                        int((y + fy * arrow_gain) * upscale),
                    ),
                    colour,
                    1,
                    tipLength=0.3,
                )

        cv2.rectangle(image, (0, 0), (frame_width, 46), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"t={times[index]:5.2f}s   swing={field.omega[index]:+.3f} rad/s",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"bucket={field.v_up[index]:+.3f} L/s   speed={field.speed[index]:.3f} L/s",
            (6, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 255, 180),
            1,
        )

        panel = np.full((strip_height, frame_width, 3), 25, np.uint8)
        for curve, colour, offset, label in curves:
            points = [
                (int(i * frame_width / max(len(curve) - 1, 1)), int(offset - curve[i] * 18))
                for i in range(len(curve))
                if np.isfinite(curve[i])
            ]
            for k in range(1, len(points)):
                cv2.line(panel, points[k - 1], points[k], colour, 1)
            cv2.line(panel, (0, offset), (frame_width, offset), (70, 70, 70), 1)
            cv2.putText(
                panel, label, (4, offset - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1
            )
        playhead = int(index * frame_width / max(len(frames) - 1, 1))
        cv2.line(panel, (playhead, 0), (playhead, strip_height), (255, 255, 255), 1)

        canvas = np.vstack([image, panel])
        if canvas.shape[:2] != (frame_height + strip_height, frame_width):
            raise RuntimeError(f"frame {index} is {canvas.shape[:2]}, not the writer's size")
        writer.write(canvas)
        written += 1

    writer.release()
    # A writer that silently drops every frame leaves a valid but tiny file; it
    # has happened here before, so the size is checked rather than trusted.
    if path.stat().st_size < 10_000:
        raise RuntimeError(f"{path} is {path.stat().st_size} bytes after {written} frames")
    log.info("wrote %s (%d frames)", path, written)
    return path


def _normalise(signal: np.ndarray) -> np.ndarray:
    """Scale to roughly [-1, 1] for drawing, without moving the zero line."""
    peak = np.nanmax(np.abs(signal)) if np.isfinite(signal).any() else 0.0
    return signal / peak if peak > 0 else signal
