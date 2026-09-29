"""Pass 2 of the phase search: one start time inside each phase window.

`windows.py` says roughly where a phase starts; this picks the moment inside it:

    dig    the lowest 2D speed inside the dig window (Aanya: "for now for dig just
           find the local min of speed in that interval")
    haul   when the height starts rising               (the bucket starts lifting)
    dump   when the aspect ratio starts falling        (the bucket starts tipping)
    swing  when it starts moving along the pile-truck line, either way (the swing begins)

Each of the last three starts at the fastest movement in that direction inside the
window, then walks back while the rate is still moving that way faster than its own
noise (3x the noise measured in the window). The start is always inside the window
(Aanya: "the specific frame has to be within the interval"); if the movement is still
going at the window's start, the start is the window's start. When nothing in the
window moves faster than its noise, the start is the window's centre, flagged (Aanya:
"it needs to report a frame in the window so either it finds one or just the center
of the window").

Dumping starts at TIPPING (the aspect ratio falling), not when the bucket arrives over
the truck: the task's wording is "starts tipping or uncurling".

No labels are read to find anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cues
from .features import FeatureTable, Scene
from .rates import noise_scale
from .windows import from_found

SIGMA = 3.0  # "moving" = beyond 3x the rate's own noise

# phase -> (the rate, the direction that phase moves it in (0: either))
RATE = {
    "hauling": ("dh_dt", +1),
    "dumping": ("aspect_ratio_dt", -1),
    # either way: on the long clip the bucket first moves PAST the truck, away from
    # the pile (the open "swing overshoot" question), so no direction is assumed
    "swinging": ("pile_truck_pos_dt", 0),
}


@dataclass(frozen=True)
class PhaseStart:
    """One phase's start, in seconds from the start of the video."""

    phase: str
    time: float
    window: tuple[float, float]  # the interval pass 1 found; ``time`` is inside it
    fallback: bool  # nothing moved faster than noise in the window: its centre is used
    low_agreement: bool  # pass 1 found no majority: the window is where most weight agrees


@dataclass(frozen=True)
class PhaseSearch:
    """Everything the phase search found, in the order it found it."""

    starts: list[PhaseStart]
    gaps: list[dict]  # cycles abandoned because a stage was missed
    stop: dict | None  # why the search ended early, or None if it reached the clip's end


def movement_start(rate: np.ndarray, t: np.ndarray, window, direction: int) -> float | None:
    """Walk back from the fastest movement in the window to where it began.

    "Began" = the rate, signed so this phase's movement is positive, was last no
    faster than its own noise band. Why not "back to rest": the bucket changes shape
    and height all through the haul, so the tip begins out of motion, not out of
    stillness -- a walk that waits for rest slid back 4-10 s into the haul on two of
    three dumps.
    """
    signed = np.abs(rate) if direction == 0 else rate * direction
    inside = np.where((t >= window[0]) & (t <= window[1]) & np.isfinite(signed))[0]
    if len(inside) < 5:
        return None
    band = SIGMA * noise_scale(signed[inside])
    i = inside[np.argmax(signed[inside])]
    if not np.isfinite(band) or signed[i] <= band:
        return None  # nothing in the window moves this way faster than noise
    while i > inside[0] and np.isfinite(signed[i - 1]) and signed[i - 1] > band:
        i -= 1
    return float(t[i])


def pick_starts(
    t: np.ndarray, F: dict[str, np.ndarray], steps: list[dict]
) -> list[PhaseStart]:
    """One start per window, always inside that window. Steps with no window are skipped."""
    out = []
    for s in steps:
        w = s["window"]
        if w is None:
            continue
        if s["phase"] == "digging":
            inside = np.where((t >= w[0]) & (t <= w[1]) & np.isfinite(F["speed_2d"]))[0]
            at = float(t[inside[np.argmin(F["speed_2d"][inside])]]) if len(inside) else None
        else:
            key, direction = RATE[s["phase"]]
            at = movement_start(F[key], t, w, direction)
        fallback = at is None
        out.append(
            PhaseStart(
                phase=s["phase"],
                time=(w[0] + w[1]) / 2 if fallback else at,
                window=(float(w[0]), float(w[1])),
                fallback=fallback,
                low_agreement=bool(s.get("weak")),
            )
        )
    return out


def find_phase_starts(table: FeatureTable, scene: Scene) -> PhaseSearch:
    """Run both passes over a feature table. Raises ``cues.NoPhases`` when the video
    lacks what the method needs (no truck, no pile)."""
    t, F = cues.signals(table, scene)
    steps, stop, gaps = from_found(t, F, cues.truck_side(F))
    return PhaseSearch(starts=pick_starts(t, F, steps), gaps=gaps, stop=stop)
