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
