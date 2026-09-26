"""Feature graphs.

The rule this stage works to: a human must be able to see the four phase
boundaries in these signals *before* any rule is asked to find them. If a
boundary is not visible by eye in some panel here, no threshold is going to
recover it, and the honest move is to find a better feature rather than a
cleverer detector.

Every panel carries a note saying which transition it is meant to serve, so a
reader can check the claim rather than take it. Panels with no transition named
are context: they explain what the machine is doing, and exist to make a wrong
call visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a cluster, and none needed
import matplotlib.pyplot as plt
import numpy as np

from .features import FeatureTable, Scene
from .logging_setup import get_logger

log = get_logger(__name__)

_INK = "#e8edf5"
_DIM = "#a8b8cc"
_AXIS = "#70808f"
_GROUND = "#0b0f16"
_EDGE = "#2a3341"
_ZERO = "#54606f"
_TRACE = "#e8edf5"


def _panels(table: FeatureTable) -> list[tuple[str, np.ndarray, float | None]]:
    """(label, values, where-to-draw-a-zero-line).

    Ordered to read top to bottom as the machine's story: where the bucket is,
    how fast it is going, where it is relative to the two things that matter,
    and what shape it presents.
    """
    return [
        ("bucket x  (L)", table.bucket_x, None),
        ("bucket y  (L)\nimage coords: down is +", table.bucket_y, None),
        ("height above slew centre  (L)\nup is +    -> T1, T2", table.height, None),
        ("dh/dt  (L/s)\n-> T1: arrives at 0", table.dh_dt, 0.0),
        ("d2h/dt2  (L/s^2)\n-> T2: arrives at 0", table.d2h_dt2, 0.0),
        ("dx/dt  (L/s)", table.dx_dt, 0.0),
        ("|dx/dt|  (L/s)\n-> T4: departs 0", table.speed_x, 0.0),
        ("bucket - cabin  x  (L)", table.rel_cabin_x, 0.0),
        ("bucket - cabin  y  (L)\n+ = bucket above cabin", table.rel_cabin_y, 0.0),
        ("bucket - truck  x  (L)", table.rel_truck_x, 0.0),
        ("bucket ^ truck box\n-> T3 gate", table.truck_overlap, None),
        ("box aspect ratio  w/h\n-> T3: peaks as it tips", table.aspect_ratio, None),
        ("radius from slew centre  (L)", table.radius, None),
    ]


def plot_features(
    table: FeatureTable,
    scene: Scene,
    path: str | Path,
    onsets: dict[str, float] | None = None,
) -> Path:
    """Every feature against time, stacked on a shared axis.

    Args:
        onsets: detected phase onsets in seconds, drawn as vertical lines. Absent
            until the state machine exists, and the graphs are useful without it
            -- that is the point of looking at them first.
    """
    path = Path(path)
    panels = _panels(table)
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(13, 1.9 * len(panels)), dpi=110, sharex=True
    )
    figure.patch.set_facecolor(_GROUND)

    colours = {
        "digging": "#5ab4f0",
        "hauling": "#78dc78",
        "dumping": "#f0aa46",
        "swinging": "#dc82f0",
    }

    for axis, (label, values, zero) in zip(axes, panels, strict=True):
        axis.set_facecolor(_GROUND)
        axis.plot(table.time_seconds, values, color=_TRACE, lw=1.2)
        if zero is not None:
            axis.axhline(zero, color=_ZERO, lw=0.8)
        _shade_gaps(axis, table)
        for name, when in (onsets or {}).items():
            if when is not None:
                axis.axvline(when, color=colours.get(name, _DIM), lw=1.6)
        axis.set_ylabel(label, color=_DIM, fontsize=8)
        axis.tick_params(colors=_AXIS, labelsize=7)
        for spine in axis.spines.values():
            spine.set_color(_EDGE)

    caption = "seconds"
    if onsets:
        caption += "    solid vertical = detected onset"
    axes[-1].set_xlabel(caption, color=_DIM)
    axes[0].set_title(
        f"Box features.  L = {scene.scale:.0f} px, "
        f"slew centre ({scene.pivot[0]:.0f}, {scene.pivot[1]:.0f}) px, "
        f"truck {'found' if scene.truck_box else 'not found'}.  No optical flow.",
        color=_INK,
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=figure.get_facecolor())
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def _shade_gaps(axis, table: FeatureTable) -> None:
    """Mark where the bucket had no mask.

    Drawn rather than left blank because a flat line through a gap and a flat
    line through a stationary bucket look identical, and only one of them is a
    measurement.
    """
    missing = ~table.found
    if not missing.any():
        return
    times = table.time_seconds
    start = None
    for index, gone in enumerate(missing):
        if gone and start is None:
            start = index
        elif not gone and start is not None:
            axis.axvspan(times[start], times[index - 1], color="#3a2020", alpha=0.6, lw=0)
            start = None
    if start is not None:
        axis.axvspan(times[start], times[-1], color="#3a2020", alpha=0.6, lw=0)
