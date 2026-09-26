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
from typing import Any

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



def _normalise(signal: np.ndarray) -> np.ndarray:
    """Scale to roughly [-1, 1] for drawing, without moving the zero line."""
    peak = np.nanmax(np.abs(signal)) if np.isfinite(signal).any() else 0.0
    return signal / peak if peak > 0 else signal


def plot_mask_contact_sheet(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    times: np.ndarray,
    path: str | Path,
    tiles: int = 16,
    columns: int = 4,
    upscale: int = 3,
) -> Path:
    """A grid of frames spanning the clip, each with its mask filled in.

    The point is to judge segmentation over the WHOLE video rather than at a few
    chosen moments: a mask that drifts, or that swallows the truck, shows up as
    one bad tile among good ones. `frames` and `masks` are parallel and in sample
    order -- the same join the motion stage insists on, for the same reason.
    """
    if len(frames) != len(masks):
        raise ValueError(f"{len(frames)} frames but {len(masks)} masks; they must be parallel")

    path = Path(path)
    picks = np.linspace(0, len(frames) - 1, tiles).astype(int)
    panels = []
    for index in picks:
        image = cv2.resize(
            frames[index], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST
        )
        mask = cv2.resize(
            masks[index].astype(np.uint8),
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        tinted = image.copy()
        tinted[mask] = (0, 0, 255)
        image = cv2.addWeighted(image, 0.55, tinted, 0.45, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
        cv2.rectangle(image, (0, 0), (image.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"t={times[index]:5.2f}s   mask={masks[index].mean() * 100:.1f}% of frame",
            (5, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        panels.append(image)

    rows = [
        np.hstack(panels[i : i + columns])
        for i in range(0, len(panels) - columns + 1, columns)
    ]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info("wrote %s", path)
    return path


def plot_kinematic_signals(
    signals: Any,
    path: str | Path,
    marks: dict[str, float] | None = None,
    truck_gate: float = 0.05,
) -> Path:
    """Four panels, one per signal, with the moment each one is meant to find.

    ``marks`` is an optional ``{label: seconds}`` overlay -- hand-marked
    boundaries, a previous run's onsets, anything. It is a caller's argument on
    purpose: the pipeline must never read the evaluation ground truth, and a
    plotting function that reached for it would quietly break that. A test
    greps this package to keep it honest.

    Raw and smoothed are both drawn. The smoothing is Savitzky-Golay, which is
    symmetric and therefore zero-phase -- it does not shift a departure or an
    extremum in time. A causal filter would lag every boundary by the same
    amount, which is exactly the systematic error a +/-0.6 s tolerance cannot
    absorb, so it is worth seeing that the thin and thick lines turn at the
    same place.
    """
    path = Path(path)
    times = np.asarray(signals.time_seconds)

    panels = [
        (
            "omega  (rad/s)\nhouse vs stick vs bucket",
            [
                ("house", signals.omega_house, "#c62828"),
                ("stick", signals.omega_stick, "#1f4e79"),
                ("bucket", signals.omega_bucket, "#2e7d32"),
            ],
            0.0,
            "T4 swinging: the HOUSE departs rest",
        ),
        (
            "delta omega  (rad/s)\nbucket relative to stick",
            [("bucket - stick", signals.delta_omega, "#6a1b9a")],
            0.0,
            "T3 dumping: departs zero WHILE over the truck",
        ),
        (
            "h  (units of L)\nbucket height above the pivot",
            [("h", signals.h, "#ef6c00")],
            None,
            "T1 / T2: arrives at, then departs, its own plateau",
        ),
        (
            "bucket over truck\nfraction of bucket inside the bed",
            [("overlap", signals.truck_overlap, "#00695c")],
            truck_gate,
            "the GATE for T3 -- a location, never a clock",
        ),
    ]

    figure, axes = plt.subplots(
        len(panels), 1, figsize=(15, 2.6 * len(panels)), sharex=True
    )
    for axis, (label, series, reference, note) in zip(axes, panels, strict=True):
        for name, values, colour in series:
            values = np.asarray(values, dtype=np.float64)
            axis.plot(times, values, linewidth=0.8, alpha=0.35, color=colour)
            axis.plot(
                times,
                _smooth_for_display(values),
                linewidth=2.0,
                color=colour,
                label=name,
            )
        if reference is not None:
            axis.axhline(reference, color="#999999", linewidth=0.9, linestyle="--")
        axis.set_ylabel(label, fontsize=9)
        axis.grid(alpha=0.18)
        axis.text(
            0.995,
            0.06,
            note,
            transform=axis.transAxes,
            fontsize=8,
            color="#555555",
            ha="right",
        )
        if len(series) > 1:
            axis.legend(fontsize=8, loc="upper left", ncol=len(series))

        if marks:
            for name, when in marks.items():
                axis.axvline(when, color="#b71c1c", linewidth=1.0, alpha=0.55)
                if axis is axes[0]:
                    axis.text(
                        when,
                        axis.get_ylim()[1],
                        f" {name}",
                        rotation=90,
                        fontsize=7.5,
                        color="#b71c1c",
                        va="top",
                    )

    axes[-1].set_xlabel("seconds")
    figure.suptitle(
        "Kinematic signals -- each panel is the evidence for one onset", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    log.info("wrote %s", path)
    return path
