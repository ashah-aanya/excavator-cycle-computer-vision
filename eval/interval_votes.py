"""Phase windows: each graph's event, widened by its own timing uncertainty, then voted.

**Evaluation scaffolding, not pipeline code.** Imports nothing from the pipeline.

For each phase, three graphs each look for their own event (the end of a big drop,
a knee, the bottom of a dip with a spike after ...). Walking the clip in phase
order, a phase is searched for only after the previous phase has started:

1. each graph finds its FIRST event after the search start (haul's events must
   also pass its hard gates);
2. the event has a CENTRE -- where that graph says the phase starts;
3. the centre is widened into a window by that graph's timing uncertainty there:

       half-width = noise of the signal / |slope after - slope before|

   with the two slopes fitted over ``SLOPE_SIDE`` s either side of the centre. The
   events are all turning points (a corner, the end of a drop, a peak), where the
   slope itself is about zero; what pins a turning point down is how sharply the
   slope CHANGES there. A sharp corner gets a tight window, a rounded one a wide one;
4. the half-width is clamped: at least ``MIN_HALF`` (``RATE_MIN_HALF`` for rate
   signals, whose 0.9 s smoothing blurs any event by about that much) and at most
   ``MAX_HALF``. Equal on both sides. A window is cut at the search start and the
   width is not moved to the other side. A window that hit the maximum is flagged;
5. the phase's window is where at least 2 of the 3 windows overlap.

The next phase's search starts at the start of this phase's window (``--anchor
chain``) or at the hand-marked start of this phase (``--anchor marks``). Every
number drawn from the output is computed here; the marks are loaded only to be
shown and, with ``--anchor marks``, to set where each search starts.

Usage::

    uv run python eval/interval_votes.py --clip long --anchor marks --out votes.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent
ORDER = ("digging", "hauling", "dumping", "swinging")

# Window half-widths, in seconds. Minimum +/-0.4 s (0.8 s wide) and maximum +/-0.6 s
# (1.2 s wide) -- Aanya, 2026-09-27; the minimum was first 0.25 s. Rate signals are
# smoothed over 0.9 s, so their events cannot be placed more tightly than about
# +/-0.45 s, which overrides the 0.4 s floor.
MIN_HALF = 0.5  # broadened from 0.4 (Aanya: "can we broaden the windows")
RATE_MIN_HALF = 0.5
MAX_HALF = 0.8  # broadened from 0.6
SLOPE_SIDE = 1.0  # seconds either side of an event for the slope-change estimate
RATES = {"dx_dt", "dh_dt", "speed_2d", "d2x_dt2", "speed_x"}
GATE_LOOKAHEAD = 1.0  # s after a haul event in which its gates may be met
HAUL_HEIGHT_SECONDS = 1.5  # "a second long or something", broadened from 1.0
STEEP = 0.25  # a 2 s line moving >= this fraction of the spread is a steep rise
CLIP_START_SECONDS = 5.0  # how far into a clip the opening-dig rule may look


def _check_cues():
    spec = importlib.util.spec_from_file_location("check_cues", EVAL / "check_cues.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _check_cues()


# ------------------------------------------------------------------ uncertainty


def noise_scale(values: np.ndarray) -> float:
    """How much the signal jitters sample to sample: the MAD of successive
    differences, scaled to a standard deviation. Same formula as the pipeline's
    ``onsets.noise_scale``, re-implemented so this file needs no pipeline import."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    steps = np.diff(v)
    return float(np.median(np.abs(steps - np.median(steps))) * 1.4826 / np.sqrt(2.0))


def half_width(
    values: np.ndarray, times: np.ndarray, centre: float, key: str, min_half=None
) -> tuple:
    """Timing uncertainty at ``centre``: noise / |slope change|, clamped.
    Returns (half-width, raw estimate, hit_max)."""
    noise = noise_scale(values)

    def slope(lo, hi):
        m = (times >= lo) & (times <= hi) & np.isfinite(values)
        if m.sum() < 3:
            return np.nan
        return float(np.polyfit(times[m], values[m], 1)[0])

    change = abs(slope(centre, centre + SLOPE_SIDE) - slope(centre - SLOPE_SIDE, centre))
    raw = noise / change if np.isfinite(change) and change > 0 else float("inf")
    floor = max(RATE_MIN_HALF if key in RATES else MIN_HALF, min_half or 0.0)
    half = min(max(raw, floor), MAX_HALF)
    return half, raw, raw >= MAX_HALF


# ---------------------------------------------------------------------- finders


def shape_runs(values, times, shape):
    """Spans where the 2 s shape reading is ``shape``."""
    return cc._runs(cc.hits(cc.shapes(values, times), shape), times)


def _i(t, s):
    return int(np.searchsorted(t, s))


def _walk_down(v, i, tol):
    """From ``i``, follow the signal down while it keeps falling (within ``tol`` of
    noise); return the index of the lowest point reached -- the minimum just before
    the signal starts going up again."""
    lo = i
    while i + 1 < len(v) and np.isfinite(v[i + 1]) and v[i + 1] <= v[i] + tol:
        i += 1
        if v[i] < v[lo]:
            lo = i
    return lo


def finders(t, F, side):
    """Three graphs per phase. Each is a dict:

    * ``id``, ``key`` (the signal), ``values`` (signed: horizontal signals are +
      toward the truck), ``label``;
    * ``mode``: "centre" -- the window is the event's centre +/- its timing
      uncertainty -- or "span" -- the window is the event's own span (a supporting
      graph that says WHERE, not exactly when);
    * ``min_half``: a per-graph minimum half-width, where Aanya asked for a wider one;
    * ``find(anchor, cycle)``: the first event after ``anchor`` as
      {"event": (a, b), "centre": c, "context": (a, b) or None}, or None. ``cycle``
      carries what has been found so far in this cycle (the dig window);
    * ``background``: every candidate event in the clip, for drawing only.
    """
    h = F["height"]
    speed = F["speed_2d"]
    dx = F["dx_dt"] * side
    x = F["bucket_x"] * side
    rel_cabin_x = F["rel_cabin_x"] * side
    rel_truck_x = F["rel_truck_x"] * side
    overlap = F["truck_overlap"]

    humps = cc.excursion_windows(speed, t, "peak", min_size=0.5)
    dx_dips = cc.excursion_windows(dx, t, "dip", min_size=0.5)
    h_drops = cc.big_drops(h, t, min_drop=0.5, max_seconds=6.0)
    aspect_drops = cc.big_drops(F["aspect_ratio"], t)
    spread = np.nanpercentile(h, 95) - np.nanpercentile(h, 5)
    from scipy.signal import find_peaks

    h_peaks = find_peaks(h, prominence=0.03 * spread)[0]
    dx_noise = noise_scale(dx)

    # ---- dig
    def speed_min(anchor, _cycle):
        hump = next((d for d in humps if d["apex"] >= anchor), None)
        if hump is None:
            return None
        # the FIRST minimum after the hump: stop as soon as speed starts going up. A
        # noise tolerance walked over a small bump at 78.8 s to a later, lower minimum
        # at 80.0 s, past the dig ("it missed the first hump").
        m = _walk_down(speed, _i(t, hump["apex"]), 0.0)
        return {
            "event": (hump["apex"], float(t[m])),
            "centre": float(t[m]),
            "context": (hump["start"], hump["end"]),
        }

    def height_drop_end(anchor, _cycle):
        d = next((d for d in h_drops if d["bottom"] >= anchor), None)
        if d is None:
            return None
        return {
            "event": (d["onset"], d["bottom"]),
            "centre": d["bottom"],
            "context": (d["onset"], d["bottom"]),
        }

    def dx_crossing(anchor, _cycle):
        dip = next((d for d in dx_dips if d["apex"] >= anchor), None)
        if dip is None:
            return None
        # "crosses the x-axis" = reaches 0 WITHIN THE SIGNAL'S NOISE: after the 50 s
        # dip, dx/dt comes back to -0.005 -- the first hump -- without going above 0,
        # and a strict crossing skipped it for the next hump at 51.6 s.
        k = _i(t, dip["apex"])
        up = next((j for j in range(k, len(t)) if dx[j] >= -dx_noise), None)
        if up is None:
            return None
        peak = up
        while peak + 1 < len(t) and dx[peak + 1] >= dx[peak]:
            peak += 1  # the little peak
        end = _walk_down(dx, peak, 0.0)  # "then the drop again that happens right after"
        return {
            "event": (float(t[up]), float(t[end])),
            "centre": float(t[up]),
            "context": (dip["start"], dip["end"]),
        }

    # ---- haul (gated per cycle, see haul_gate)
    dip_spike = cc.dip_then_rise_mask(dx, t)
    # "consider more of the knee" (Aanya). The STRICT knee picks WHICH bend (the sharp
    # one); a LOOSER knee -- before moves < 2x the flat limit, after rises >= half the
    # steep limit -- measures how far that bend extends. The loose one alone also
    # matched a smaller bend mid-dig.
    takeoff_tx = cc.knee_mask(rel_truck_x, t, "rising", "flat_to_steep") & (rel_truck_x < 0)
    bend_tx = cc._runs(
        cc.knee_mask(
            rel_truck_x,
            t,
            "rising",
            "flat_to_steep",
            flat_max=2 * cc.FLAT,
            steep_min=STEEP / 2,
        )
        & (rel_truck_x < 0),
        t,
    )

    def gated_first(mask, anchor, cycle):
        """The first event after ``anchor`` that passes the haul gates somewhere in
        itself or the second after it. Checked at the event's single first instant,
        the gate skipped the right event: the bottom of the dx/dt dip comes a moment
        BEFORE the bucket rises above its dig height -- the lift is the spike after."""
        gate = haul_gate(t, F, anchor, cycle)
        for a, b in cc._runs(mask, t):
            if b < anchor:
                continue
            a = max(a, anchor)
            if gate[_i(t, a) : _i(t, b + GATE_LOOKAHEAD) + 1].any():
                return (a, b)
        return None

    _, h_after = cc.side_changes(h, t, cc.SIDE)

    def haul_height(_anchor, cycle):
        """Aanya: "draw a line from the tallest height in dig window to the graph for
        height and then the window starts there with the leftmost bar of the window
        there and you can make the window a second long". The first time, after the
        dig window, that height rises above the tallest height inside it."""
        w = cycle.get("dig_window")
        if w is None:
            return None
        inside = (t >= w[0]) & (t <= w[1])
        level = float(np.nanmax(h[inside]))
        # "...but also make sure it keeps going generally up after that": above the
        # line AND the 2 s line ahead is rising. A scooping wiggle that pokes above
        # the line and falls back (51.0 s in cycle 3) does not count.
        start = next(
            (
                j
                for j in range(_i(t, w[1]), len(t))
                if h[j] > level and np.isfinite(h_after[j]) and h_after[j] >= cc.FLAT
            ),
            None,
        )
        if start is None:
            return None
        a = float(t[start])
        return {
            "event": (a, a + HAUL_HEIGHT_SECONDS),
            "centre": a,
            "context": None,
            "level": level,
        }

    def haul_dx_min(anchor, cycle):
        e = gated_first(dip_spike, anchor, cycle)
        if e is None:
            return None
        lo, hi = _i(t, max(e[0] - cc.SIDE, anchor)), _i(t, e[1]) + 1
        m = lo + int(np.nanargmin(dx[lo:hi]))
        return {"event": e, "centre": float(t[m]), "context": None}

    def haul_truck_x(anchor, cycle):
        e = gated_first(takeoff_tx, anchor, cycle)
        if e is not None:  # grow the sharp knee to the whole bend around it
            bend = next((r for r in bend_tx if r[0] <= e[1] and r[1] >= e[0]), None)
            if bend is not None:
                e = (max(min(e[0], bend[0]), anchor), max(e[1], bend[1]))
        return None if e is None else {"event": e, "centre": sum(e) / 2, "context": None}

    # ---- dump
    # Aanya (2026-09-27): "The window for dump should include the steepest drops in
    # aspect ratio. That signal definitely will be a steep drop, but it needs to be
    # very validated by some of the other things that we're seeing there. I think
    # relative cabin x position should be going up." Her key for bucket - cabin x at
    # the dump: "down bump and now its going up again". For height: "It's the second
    # hump that it needs to contain. There's a dip and then a second hump, and it
    # should contain it on the going-up part of that hump. It can find the maximum of
    # the second hump and then backtrack a little bit to make the interval."
    _, cabin_after = cc.side_changes(rel_cabin_x, t, cc.SIDE)
    cabin_noise = noise_scale(rel_cabin_x)
    with np.errstate(invalid="ignore"):
        cabin_steep = cabin_after >= STEEP  # the 2 s ahead rises by >= a quarter of the spread

    def cabin_down_bump(anchor):
        """The haul carries the bucket toward the truck: a steep rise in bucket - cabin
        x. After it: a down-bump, then going up again, until the return swing's steep
        rise. Returns (index of the down-bump's minimum, index where the swing's steep
        rise starts), or None."""
        i = _i(t, anchor)
        while i < len(t) and not cabin_steep[i]:
            i += 1  # to the haul's steep rise
        while i < len(t) and cabin_steep[i]:
            i += 1  # past it
        e = i
        while e < len(t) and not cabin_steep[e]:
            e += 1  # to the swing's steep rise
        if e >= len(t) or e - i < 2:
            return None
        # The haul's rise is not always one unbroken steep stretch, so "past it" can
        # stop partway up. Walk on up to the first top (the hump), then down to the
        # bottom of the down-bump -- each within the signal's noise, so a wiggle does
        # not stop the walk. Not the highest point overall: bucket - cabin x keeps
        # climbing into the swing, so that would be the swing.
        top = i
        while top + 1 < e and rel_cabin_x[top + 1] >= rel_cabin_x[top] - cabin_noise:
            top += 1
        dip = top
        while dip + 1 < e and rel_cabin_x[dip + 1] <= rel_cabin_x[dip] + cabin_noise:
            dip += 1
        lo = top + int(np.nanargmin(rel_cabin_x[top : dip + 1]))
        return (lo, e) if e - lo >= 2 else None

    def cabin_x_going_up(anchor, _cycle):
        found = cabin_down_bump(anchor)
        if found is None:
            return None
        dip, e = found
        return {
            "event": (float(t[dip]), float(t[e])),
            "centre": float(t[dip]),
            "context": None,
        }

    def aspect_steepest(anchor, _cycle):
        """The first big aspect-ratio drop whose steepest point comes after bucket -
        cabin x's down-bump: validated by it. A drop before it -- such as the
        tracking glitch mid-haul -- is not the tip."""
        found = cabin_down_bump(anchor)
        after = float(t[found[0]]) if found else anchor
        d = next((d for d in aspect_drops if d["steepest"] >= after), None)
        if d is None:
            return None
        return {
            "event": (d["onset"], d["bottom"]),
            "centre": d["steepest"],
            "context": (d["onset"], d["bottom"]),
        }

    def height_second_hump(anchor, _cycle):
        """The second hump is the peak just before the next big height drop (the
        bucket going down to the pile). From its maximum, backtrack down the climb
        to where the climb begins ("backtrack a little bit"): the point where 90% of
        the hump's rise -- from the lowest point since the previous peak up to the
        maximum -- is undone. That is the dip when there is one ("There's a dip and
        then a second hump") and the end of the flat when height sits flat instead.
        The interval is that point -> the maximum."""
        drop = next((d for d in h_drops if d["onset"] >= anchor), None)
        if drop is None:
            return None
        before = [p for p in h_peaks if anchor <= t[p] <= drop["onset"] + 1e-9]
        if not before:
            return None
        top = before[-1]
        prev = [p for p in h_peaks if p < top]
        lo = max(prev[-1] if prev else 0, _i(t, anchor))
        base = float(np.nanmin(h[lo : top + 1]))
        level = base + 0.1 * (h[top] - base)  # 90% of the rise undone
        q = top
        while q > lo and h[q] > level:
            q -= 1
        return {"event": (float(t[q]), float(t[top])), "centre": float(t[q]), "context": None}

    # ---- swing (unchanged)
    x_takeoff = cc.knee_windows(x, t, "rising", "flat_to_steep")
    overlap_end = shape_runs(overlap, t, "drop ends -> flat")
    h_peak_runs = shape_runs(h, t, "peak")

    def from_spans(spans):
        def find(anchor, _cycle):
            e = first_event(spans, anchor)
            return None if e is None else {"event": e, "centre": sum(e) / 2, "context": None}

        return find

    def g(id_, key, values, label, find, mode="centre", min_half=None, background=()):
        return {
            "id": id_,
            "key": key,
            "values": values,
            "label": label,
            "find": find,
            "mode": mode,
            "min_half": min_half,
            "background": list(background),
        }

    return {
        "digging": [
            g(
                "speed_min",
                "speed_2d",
                speed,
                "2D speed: the minimum right after the big hump, before it rises again",
                speed_min,
                background=[(d["start"], d["end"]) for d in humps],
            ),
            g(
                "height_drop_end",
                "height",
                h,
                "height: the end of the big drop",
                height_drop_end,
                background=[(d["onset"], d["bottom"]) for d in h_drops],
            ),
            g(
                "dx_crossing",
                "dx_dt",
                dx,
                "dx/dt (toward truck +): after the big dip, from crossing 0 upward, over the "
                "little peak, to crossing 0 downward",
                dx_crossing,
                mode="span",
                background=[(d["start"], d["end"]) for d in dx_dips],
            ),
        ],
        "hauling": [
            g(
                "height_takeoff",
                "height",
                h,
                "height: where it first rises above the tallest height in the dig window; "
                "1 s from there",
                haul_height,
                mode="fixed",
            ),
            g(
                "dx_min",
                "dx_dt",
                dx,
                "dx/dt (toward truck +): the minimum after the plateau, before the spike",
                haul_dx_min,
                background=cc._runs(dip_spike, t),
            ),
            g(
                "truck_x_takeoff",
                "rel_truck_x",
                rel_truck_x,
                "bucket - truck x (toward truck +): the whole knee bend, flat -> rising, "
                "while negative",
                haul_truck_x,
                mode="span",
                min_half=0.5,
                background=cc._runs(takeoff_tx, t),
            ),
        ],
        "dumping": [
            g(
                "aspect_steepest",
                "aspect_ratio",
                F["aspect_ratio"],
                "box aspect ratio: the steepest part of the first big drop after bucket - "
                "cabin x's down-bump",
                aspect_steepest,
                background=[(d["onset"], d["bottom"]) for d in aspect_drops],
            ),
            g(
                "cabin_x_going_up",
                "rel_cabin_x",
                rel_cabin_x,
                "bucket - cabin x (toward truck +): from the down-bump's minimum, going up, "
                "until the swing's steep rise",
                cabin_x_going_up,
                mode="span",
                background=cc._runs(cabin_steep, t),
            ),
            g(
                "height_second_hump",
                "height",
                h,
                "height: the second hump's going-up part -- backtracked from its maximum "
                "to where its climb begins",
                height_second_hump,
                mode="span",
                background=[(float(t[p]) - 0.1, float(t[p]) + 0.1) for p in h_peaks],
            ),
        ],
        "swinging": [
            g(
                "x_takeoff",
                "bucket_x",
                x,
                "bucket x (toward truck +): knee, flat -> steep rise",
                from_spans(x_takeoff),
                background=x_takeoff,
            ),
            g(
                "overlap_end",
                "truck_overlap",
                overlap,
                "bucket ^ truck box: drop ends -> flat (overlap over)",
                from_spans(overlap_end),
                background=overlap_end,
            ),
            g(
                "height_peak",
                "height",
                h,
                "height: peak",
                from_spans(h_peak_runs),
                background=h_peak_runs,
            ),
        ],
    }


def haul_gate(t, F, anchor, cycle):
    """Hard gates on haul: above the height of THIS cycle's dig window (its median
    height; the running median since the dig started when no dig window was found)
    and no truck overlap."""
    w = cycle.get("dig_window")
    if w is not None:
        inside = (t >= w[0]) & (t <= w[1])
        ref = np.full(len(t), float(np.nanmedian(F["height"][inside])))
    else:
        ref = cc._running_median(F["height"], int(np.searchsorted(t, anchor)))
    with np.errstate(invalid="ignore"):
        return (F["height"] > ref) & (F["truck_overlap"] <= 0)


def clip_start_dig(t, F, side):
    """The first dig when the clip opens mid-dig, from WEAK signals (there is no swing
    before it to read): the first moment, in the first few seconds, when the bucket is
    down (height < 0), on the pile side of the truck, not overlapping it, and at rest
    (2D speed within its rest band: 3 x its noise, at least 2% of its range)."""
    speed = F["speed_2d"]
    rest = max(3 * noise_scale(speed), 0.02 * float(np.nanmax(speed) - np.nanmin(speed)))
    with np.errstate(invalid="ignore"):
        ok = (
            (F["height"] < 0)
            & (F["rel_truck_x"] * side < 0)
            & (F["truck_overlap"] <= 0)
            & (speed <= rest)
            & (t <= t[0] + CLIP_START_SECONDS)
        )
    idx = np.flatnonzero(ok)
    return None if idx.size == 0 else float(t[idx[0]])


# ----------------------------------------------------------------------- search


def first_event(spans, anchor):
    """The first event that ends after ``anchor``, clipped to start no earlier."""
    for a, b in spans:
        if b >= anchor:
            return (max(a, anchor), b)
    return None


def widen(t, graph, found, anchor):
    """The graph's window: a "centre" graph's centre +/- its timing uncertainty, or
    a "span" graph's own event span."""
    a, b = found["event"]
    c = found["centre"]
    if graph["mode"] == "fixed":  # the window IS the event, exactly as specified
        return {
            **found,
            "half": None,
            "raw": None,
            "capped": False,
            "window": (max(a, anchor), b),
        }
    if graph["mode"] == "span":
        # a span's edges are blurred by smoothing like any event: pad both by the
        # graph's minimum half-width (missing before -- a 0.2 s dx/dt window)
        pad = max(
            RATE_MIN_HALF if graph["key"] in RATES else MIN_HALF, graph["min_half"] or 0.0
        )
        return {
            **found,
            "half": None,
            "raw": None,
            "capped": False,
            "window": (max(a - pad, anchor), b + pad),
        }
    half, raw, capped = half_width(graph["values"], t, c, graph["key"], graph["min_half"])
    return {
        **found,
        "half": half,
        "raw": raw,
        "capped": capped,
        "window": (max(c - half, anchor), c + half),
    }


def vote(t, windows, anchor):
    """Where at least 2 windows overlap: the first such stretch at or after ``anchor``."""
    count = np.zeros(len(t), int)
    for w in windows:
        if w is not None:
            a, b = w["window"]
            count += (t >= a - 1e-9) & (t <= b + 1e-9)
    runs = cc._runs((count >= 2) & (t >= anchor - 1e-9), t)
    if not runs:
        return None, 0
    a, b = runs[0]
    return (a, b), int(count[(t >= a) & (t <= b)].max())


def required(t, windows, anchor):
    """Haul: height is necessary (Aanya: "height is a requirement so make that like a
    necessary"). The window is where height's window overlaps at least one of the
    other haul graphs' windows -- the first such stretch. No height, no window."""
    height, others = windows[0], [w for w in windows[1:] if w is not None]
    if height is None or not others:
        return None, 0
    a, b = height["window"]
    inside = (t >= a - 1e-9) & (t <= b + 1e-9)
    count = np.zeros(len(t), int)
    for w in others:
        count += (t >= w["window"][0] - 1e-9) & (t <= w["window"][1] + 1e-9)
    runs = cc._runs(inside & (count >= 1) & (t >= anchor - 1e-9), t)
    if not runs:
        return None, 1
    lo, hi = runs[0]
    return (lo, hi), 1 + int(count[(t >= lo) & (t <= hi)].max())


def validated(windows):
    """Dump: the aspect ratio's window IS the window -- "the window for dump should
    include the steepest drops in aspect ratio" -- accepted only when at least one of
    the supporting graphs overlaps it: "it needs to be very validated by some of the
    other things". The supporting graphs are wide, so letting them vote with each
    other would give a window that does not depend on the aspect ratio at all."""
    primary, support = windows[0], windows[1:]
    if primary is None:
        return None, 0
    a, b = primary["window"]
    n = 1 + sum(w is not None and w["window"][0] <= b and w["window"][1] >= a for w in support)
    return ((a, b), n) if n >= 2 else (None, n)


def step(t, F, side, found, phase, anchor, cycle):
    if phase == "digging" and anchor <= t[0] + 1e-9:
        c = clip_start_dig(t, F, side)
        if c is not None:
            window = (max(c - MIN_HALF, t[0]), c + MIN_HALF)
            return {
                "phase": phase,
                "search_from": anchor,
                "graphs": [None, None, None],
                "window": window,
                "votes": 0,
                "agreed": True,
                "start": window[0],
                "clip_start": c,
            }
    graphs = found[phase]
    events = [g["find"](anchor, cycle) for g in graphs]
    if all(e is None for e in events):
        return None
    windows = [
        None if e is None else widen(t, g, e, anchor)
        for g, e in zip(graphs, events, strict=True)
    ]
    if phase == "dumping":
        window, n = validated(windows)
    elif phase == "hauling":
        window, n = required(t, windows, anchor)
    else:
        window, n = vote(t, windows, anchor)
    agreed = window is not None
    start = window[0] if agreed else min(w["window"][0] for w in windows if w is not None)
    return {
        "phase": phase,
        "search_from": anchor,
        "graphs": windows,
        "window": window,
        "votes": n,
        "agreed": agreed,
        "start": start,
    }


def from_marks(t, F, side, marks):
    """Each marked onset, searched for from the previous MARKED onset."""
    found, out, cycle = finders(t, F, side), [], {}
    for k, (phase, _) in enumerate(marks):
        if phase == "digging":
            cycle = {}
        s = step(t, F, side, found, phase, marks[k - 1][1] if k else float(t[0]), cycle)
        if s is not None:
            if phase == "digging":
                cycle["dig_window"] = s["window"]
            out.append(s)
    return found, out


def walk(t, F, side):
    """Every phase, in order, each searched for after the previous one started."""
    found, out, cycle = finders(t, F, side), [], {}
    anchor, k = float(t[0]), 0
    while True:
        phase = ORDER[k % 4]
        if phase == "digging":
            cycle = {}
        s = step(t, F, side, found, phase, anchor, cycle)
        if s is None:
            break
        if phase == "digging":
            cycle["dig_window"] = s["window"]
        out.append(s)
        if s["start"] <= anchor + 1e-9 and k > 0 and not s["agreed"]:
            break  # no progress possible
        anchor, k = s["start"] + 1e-6, k + 1
    return found, out


def _clean(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", choices=sorted(cc.CLIPS), default="long")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor", choices=("chain", "marks"), default="chain")
    args = ap.parse_args(argv)
    feat_path, label_path = cc.CLIPS[args.clip]
    t, F = cc.load_features(feat_path)
    side = cc.truck_side(F)
    marks = cc.load_onsets(label_path)
    if args.anchor == "marks":
        found, steps = from_marks(t, F, side, marks)
    else:
        found, steps = walk(t, F, side)

    data = {
        "clip": args.clip,
        "side": side,
        "anchor": args.anchor,
        "limits": {
            "min_half": MIN_HALF,
            "rate_min_half": RATE_MIN_HALF,
            "max_half": MAX_HALF,
            "slope_side": SLOPE_SIDE,
        },
        "times": [round(float(v), 3) for v in t],
        "signals": {
            g["key"]: [_clean(v) for v in g["values"]] for ph in found.values() for g in ph
        },
        "finders": {
            ph: [
                {
                    "id": g["id"],
                    "signal": g["key"],
                    "label": g["label"],
                    "mode": g["mode"],
                    "spans": g["background"],
                    "contexts": [],
                }
                for g in lst
            ]
            for ph, lst in found.items()
        },
        "steps": [
            {
                **s,
                "graphs": [
                    None if w is None else {**w, "raw": _clean(w["raw"])} for w in s["graphs"]
                ],
            }
            for s in steps
        ],
        "marks": [{"phase": p, "t": s} for p, s in marks],
    }
    args.out.write_text(json.dumps(data))
    for s in steps:
        if s["window"] is None:
            w = "no 2 agree"
        else:
            w = f"[{s['window'][0]:.2f}, {s['window'][1]:.2f}]"
        halves = " ".join(
            "  --  "
            if g is None
            else (
                " span "
                if g["half"] is None
                else f"±{g['half']:.2f}{'*' if g['capped'] else ' '}"
            )
            for g in s["graphs"]
        )
        print(f"  {s['phase']:<9} from {s['search_from']:6.2f}s  half-widths {halves}  -> {w}")
    print("  (* = hit the maximum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
