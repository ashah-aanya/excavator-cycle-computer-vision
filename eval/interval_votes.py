"""Phase windows: gates, then the required cue, then a majority of supporting cues.

**Evaluation scaffolding, not pipeline code.** Imports nothing from the pipeline.

Each phase is searched for after the hand-marked start of the previous phase (the
first dig: from the clip's start, with the clip-start rule). For each phase, in
this ORDER (Aanya, 2026-09-27):

1. GATES -- the necessary thresholds. None of the phase's cues are considered until
   all of its gates hold; cues may begin ``GATE_LOOKAHEAD`` s before that, and an
   event only counts if the gates hold somewhere in it or the second after it.
   Dig: no truck overlap, on the pile side of the truck. Haul: height above this
   cycle's dig height, no truck overlap. Dump: on the truck side of the cabin,
   height above the dig height, above the cabin. Return swing: none.
2. REQUIRED CUE -- haul: height must be part of the window. Dump: the aspect
   ratio's steepest drop IS the window.
3. SUPPORTING CUES -- a majority of the cues that found something must agree
   (overlap the window; for dump, a majority of the rest must overlap the aspect
   window). Every cue is a pattern from Aanya's key, skipping graphs that are only
   a constant shift of another.

A cue's window is its event's centre widened by its timing uncertainty (noise /
slope change), clamped to ``MIN_HALF``..``MAX_HALF``; a span cue is its span padded
by the minimum; a "check" cue (dump) is a condition read inside the aspect window.
Every number drawn from the output is computed here; the marks are loaded to set
where each search starts and to be shown.

Usage::

    uv run python eval/interval_votes.py --clip long --out votes.json
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
# Broadened again (Aanya: wider is safer once several cues validate a window --
# missing the transition cannot be undone). The maximum stays under the shortest
# phase (a dump, about 2.5 s), so no window can hold two phase starts.
# EXPERIMENT, off unless --dump-reweight: over the truck votes 3 for a dump, read
# at the tip and only in the first DUMP_VISIT_EARLY of the visit; dh/dt near 0
# votes 0.5 (Aanya: "vote the overlap higher and the dh/dt a lot weaker").
DUMP_REWEIGHT = {"on": False}
DUMP_VISIT_EARLY = 0.7  # chosen from the data: real tips at <= 65% of a visit
MIN_HALF = 0.75
RATE_MIN_HALF = 0.75
MAX_HALF = 1.2
SLOPE_SIDE = 1.0  # seconds either side of an event for the slope-change estimate
RATES = {"dx_dt", "dh_dt", "speed_2d", "d2x_dt2", "speed_x", "pile_truck_pos_dt"}
GATE_LOOKAHEAD = 1.0  # s after a haul event in which its gates may be met
HAUL_HEIGHT_SECONDS = 1.5  # "a second long or something", broadened from 1.0
NEAR = 5.0  # s: a haul cue further than this from height's window is not counted
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
    if HORIZONTAL["mode"] == "distance":
        # "let's try with making it distance to truck": the bucket's straight-line
        # image distance from the truck's centre (in the arm lengths L the features use),
        # negated so + is still "toward the truck". It shrinks on approach whichever
        # image direction the approach runs in, where x assumes a side-on camera.
        # ASSUMPTION (Aanya): the dump target is a truck -- a dump onto a pile or into
        # a hopper would break this, and most of the rules with it.
        x = -F["truck_distance"]
        dx = -F["truck_distance_dt"]
        rel_truck_x = x
    elif HORIZONTAL["mode"] == "pile_truck":
        # "the goal is to get rid of the position specific ones": where the bucket is
        # along the line from the pile to the truck (-1 pile, 0 truck, + past it), the
        # line's direction found from the video, not assumed to be image left-right
        x = F["pile_truck_pos"]
        dx = F["pile_truck_pos_dt"]
        rel_truck_x = x
    elif HORIZONTAL["mode"] == "truck_x":
        # "can you do like x distance to truck?": the signed LEFT-RIGHT gap to the
        # truck's centre. It keeps the direction (past the truck vs back toward the
        # pile) that plain distance loses.
        x = rel_truck_x
    overlap = F["truck_overlap"]

    humps = cc.excursion_windows(speed, t, "peak", min_size=0.5)
    dx_dips = cc.excursion_windows(dx, t, "dip", min_size=0.5)
    h_drops = cc.big_drops(h, t, min_drop=0.5, max_seconds=6.0)
    aspect_drops = cc.big_drops(F["aspect_ratio"], t)
    # "You can detect them multiple times, and then you can figure out which is the
    # real signal based on validating across multiple": every drop of >= 30% of the
    # range is a candidate tip; step() keeps the one the other cues support most
    aspect_candidates = cc.big_drops(F["aspect_ratio"], t, min_drop=0.3)

    def aspect_all(anchor, _cycle):
        return [
            {
                "event": (d["onset"], d["bottom"]),
                "centre": d["steepest"],
                "context": (d["onset"], d["bottom"]),
            }
            for d in aspect_candidates
            if d["steepest"] >= anchor
        ]

    spread = np.nanpercentile(h, 95) - np.nanpercentile(h, 5)
    from scipy.signal import find_peaks, peak_widths

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

    def cabin_x_going_up(anchor, cycle):
        # the down-bump is read from the phase's search start (the haul's start), not
        # from where the gates open: "skip the haul's rise" needs the rise in view
        found = cabin_down_bump(cycle.get("search_from", anchor))
        if found is None:
            return None
        dip, e = found
        return {
            "event": (float(t[dip]), float(t[e])),
            "centre": float(t[dip]),
            "context": None,
        }

    def aspect_steepest(anchor, cycle):
        """The first big aspect-ratio drop whose steepest point comes after bucket -
        cabin x's down-bump: validated by it. A drop before it -- such as the
        tracking glitch mid-haul -- is not the tip."""
        found = cabin_down_bump(cycle.get("search_from", anchor))
        after = max(float(t[found[0]]), anchor) if found else anchor
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
        # "it should be in the middle. It shouldn't contain the maximum. I was just
        # saying that you need to know that there is a maximum coming": centred where
        # height is halfway up the climb; the maximum is required but not included
        half = base + 0.5 * (h[top] - base)
        m = next((j for j in range(q, top + 1) if h[j] >= half), q)
        # the window stops where the climb is 90% of the way up -- before the rounded top
        high = base + 0.9 * (h[top] - base)
        near_top = next((j for j in range(m, top + 1) if h[j] >= high), top)
        # "make this in the middle, like it's in the valley": centred on the bottom
        # of the valley, where the climb into the second hump begins
        return {
            "event": (float(t[q]), float(t[near_top])),
            "centre": float(t[q]),
            "context": (float(t[q]), float(t[top])),
        }

    # ---- features added from Aanya's key (2026-09-27), skipping constant shifts
    radius = F["radius"]
    dh = F["dh_dt"]
    dh_noise = noise_scale(dh)
    x_drops = cc.big_drops(x, t, min_drop=0.5, max_seconds=6.0)
    x_noise = noise_scale(x)
    radius_dips = cc.excursion_windows(radius, t, "dip", min_size=0.5)

    def dig_dh(anchor, _cycle):
        """dh/dt, key: "negative approaching 0" -- after the big height drop, where
        dh/dt comes back up to 0 (within its noise)."""
        d = next((d for d in h_drops if d["bottom"] >= anchor), None)
        if d is None:
            return None
        j = next((j for j in range(_i(t, d["steepest"]), len(t)) if dh[j] >= -dh_noise), None)
        if j is None:
            return None
        return {
            "event": (d["steepest"], float(t[j])),
            "centre": float(t[j]),
            "context": (d["onset"], d["bottom"]),
        }

    def dig_radius(anchor, _cycle):
        """radius, key: "after big dip and upswing downward trend" -- after the big
        dip, the top of the upswing, where the downward trend begins."""
        dip = next((d for d in radius_dips if d["apex"] >= anchor), None)
        if dip is None:
            return None
        k = _i(t, dip["apex"])
        while k + 1 < len(t) and radius[k + 1] >= radius[k]:
            k += 1
        return {
            "event": (dip["apex"], float(t[k])),
            "centre": float(t[k]),
            "context": (dip["start"], dip["end"]),
        }

    def dig_x(anchor, _cycle):
        """bucket x, key: "just dropped a lot and now its increasing" -- and Aanya:
        the bucket's distance from the truck is valuable, and for dig bucket x carries
        it (bucket - truck x is the same curve, shifted). After the big drop the bucket
        keeps sinking slowly toward the pile, so its low point is inconsistent; what
        is consistent is ARRIVAL: the first time x is within 10% of the drop of the
        lowest it gets before rising back up. The window runs from that arrival, up to
        2 x MAX_HALF long. (The end of the steep fall was 0.9-2.0 s before every dig.)"""
        d = next((d for d in x_drops if d["bottom"] >= anchor), None)
        if d is None:
            return None
        k = _i(t, d["bottom"])
        low = k
        while low + 1 < len(t) and x[low + 1] <= x[low] + x_noise:
            low += 1  # on down, past slow sinking, to where it comes back up
        low = k + int(np.nanargmin(x[k : low + 1]))
        top = float(x[_i(t, d["onset"])])
        level = x[low] + 0.1 * (top - x[low])
        a = next((j for j in range(_i(t, d["onset"]), low + 1) if x[j] <= level), low)
        return {
            "event": (float(t[a]), min(float(t[a]) + 2 * MAX_HALF, float(t[-1]))),
            "centre": float(t[a]),
            "context": (d["onset"], d["bottom"]),
        }

    def haul_radius(anchor, _cycle):
        """radius, key: "start of dip" -- where the big V-shaped dip starts. (A knee
        from FLAT to falling found almost nothing: radius is never flat going in.)"""
        dip = next((d for d in radius_dips if d["apex"] >= anchor), None)
        if dip is None:
            return None
        return {
            "event": (dip["start"], dip["apex"]),
            "centre": dip["start"],
            "context": (dip["start"], dip["end"]),
        }

    with np.errstate(invalid="ignore"):
        dump_dh_mask = np.abs(dh) <= 3 * dh_noise  # key: "close to or at 0"
        dump_dx_mask = dx > 0  # key: "positive"
        dump_overlap_mask = overlap > 0  # key (bucket - truck x): "overlap"
        swing_dh_mask = (dh > dh_noise) & (np.gradient(dh, t) < 0)  # "positive decreasing"
    if DUMP_REWEIGHT["on"]:
        for va, vb in cc._runs(overlap > 0, t):
            dump_overlap_mask &= ~((t > va + DUMP_VISIT_EARLY * (vb - va)) & (t <= vb))
    radius_peaks = cc.excursion_windows(radius, t, "peak", min_size=0.5)
    radius_slope = np.gradient(radius, t)

    def swing_radius(anchor, _cycle):
        """radius, key: "start of bump after" -- where the STEEP part of the big bump
        begins: back from its peak while the rise is at least a quarter of its
        steepest. (Back to where it stopped falling lands seconds early: radius creeps
        up slowly before the bump.)"""
        bump = next((d for d in radius_peaks if d["apex"] >= anchor), None)
        if bump is None:
            return None
        a, p = _i(t, bump["start"]), _i(t, bump["apex"])
        if p <= a:
            return None
        steepest = float(np.nanmax(radius_slope[a:p]))
        q = p
        while q > a and radius_slope[q - 1] >= 0.25 * steepest:
            q -= 1
        return {
            "event": (float(t[q]), bump["apex"]),
            "centre": float(t[q]),
            "context": (bump["start"], bump["end"]),
        }

    def swing_dx_bump(anchor, _cycle):
        """dx/dt, key: "upward bump right after" -- the bump just before the big
        dip of the swing back: the highest dx/dt between the search start and it."""
        dip = next((d for d in dx_dips if d["apex"] >= anchor), None)
        if dip is None or dip["start"] <= anchor:
            return None
        lo, hi = _i(t, anchor), _i(t, dip["start"]) + 1
        top = lo + int(np.nanargmax(dx[lo:hi]))
        # the swing starts where dx/dt starts rising INTO the bump, not at its top:
        # walk back from the top to the low point before it (the top was 1-1.5 s late)
        k = top
        while k > lo and dx[k - 1] <= dx[k]:
            k -= 1
        return {
            "event": (float(t[k]), float(t[top])),
            "centre": float(t[k]),
            "context": (dip["start"], dip["end"]),
        }

    def first_run(mask, at="start"):
        def find(anchor, _cycle):
            e = first_event(cc._runs(mask, t), anchor)
            if e is None:
                return None
            return {
                "event": e,
                "centre": e[0] if at == "start" else sum(e) / 2,
                "context": None,
            }

        return find

    # ---- swing
    x_takeoff = cc.knee_windows(x, t, "rising", "flat_to_steep")
    # No truck-overlap cue for the swing. Aanya, 2026-09-28: "this isn't a
    # requirement for swinging. i never said it can't overlap" -- the bucket can
    # start back while still over the truck box, so "overlap ends" came mid-swing
    # and pulled the window late.
    # height's high points. Aanya: "shouldn't more heights be identified, and then
    # from that, we choose what the ideal height is? ... make this more durable for
    # understanding variations and not just looking for the most specific patterns".
    # The old cue asked for the exact shape "peak" (rising 2 s, then falling 2 s),
    # which missed rounded and plateau tops and caught a one-frame glitch. Now EVERY
    # top standing at least 15% of the spread above its surroundings is a candidate,
    # if it is at least 2 s wide at half its height (the real tops are 13-15 s wide;
    # the glitch at 65 s is 1.2 s) -- sharp, rounded and plateau tops all count.
    fps = 1.0 / float(np.median(np.diff(t)))
    h_tops, h_top_info = find_peaks(h, prominence=0.15 * spread, width=2.0 * fps)
    # each candidate's high region: the part within 20% of its top
    _, _, h_top_l, h_top_r = peak_widths(h, h_tops, rel_height=0.2)
    h_high_regions = [
        (float(np.interp(a, np.arange(len(t)), t)), float(np.interp(b, np.arange(len(t)), t)))
        for a, b in zip(h_top_l, h_top_r, strict=True)
    ]

    def swing_height(anchor, _cycle):
        """height: choose, among all the candidate tops, the LAST one before this
        cycle's big height drop (the bucket going back down to the pile). Place the
        swing where the climb into it gets within 20% of the top -- measured from
        the lowest point since the search start -- the same rule for sharp, rounded
        and plateau tops. (Within 10% of the top landed 0.6-1.3 s after her marks on
        the rounded and plateau tops; 20% landed within 0.2 s. Tuned on these two
        clips, so optimistic.)"""
        drop = next((d for d in h_drops if d["onset"] >= anchor), None)
        if drop is None:
            return None
        before = [k for k, p in enumerate(h_tops) if anchor <= t[p] <= drop["onset"] + 1e-9]
        if not before:
            return None
        k = before[-1]
        top = int(h_tops[k])
        lo = max(int(h_top_info["left_bases"][k]), _i(t, anchor))
        base = float(np.nanmin(h[lo : top + 1]))
        level = base + 0.8 * (h[top] - base)
        j = next((j for j in range(lo, top + 1) if h[j] >= level), top)
        return {
            "event": (float(t[j]), float(t[top])),
            "centre": float(t[j]),
            "context": (float(t[j]), drop["onset"]),
        }

    def from_spans(spans):
        def find(anchor, _cycle):
            e = first_event(spans, anchor)
            return None if e is None else {"event": e, "centre": sum(e) / 2, "context": None}

        return find

    def g(
        id_,
        key,
        values,
        label,
        find,
        mode="centre",
        min_half=None,
        background=(),
        role="support",
        mask=None,
        candidates=None,
        before_end=False,
    ):
        if HORIZONTAL["mode"] == "pile_truck" and key in PILE_TRUCK_NAMES:
            # name the signal the cue actually reads
            key, (old, new) = PILE_TRUCK_NAMES[key]
            label = label.replace(old, new)
        return {
            "id": id_,
            "key": key,
            "values": values,
            "label": label,
            "find": find,
            "mode": mode,
            "min_half": min_half,
            "background": list(background),
            "role": role,  # "required" (haul), "primary" (dump) or "support"
            "mask": mask,  # mode "check": a condition checked inside the primary window
            "candidates": candidates,  # primary only: every candidate event, to validate
            "before_end": before_end,  # the window must stop before the event's end
        }

    def check(id_, key, values, label, mask):
        return g(
            id_,
            key,
            values,
            label,
            None,
            mode="check",
            mask=mask,
            background=cc._runs(mask, t),
        )

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
            g(
                "dh_back_to_0",
                "dh_dt",
                dh,
                "dh/dt: negative, approaching 0 -- back to 0 after the big height drop",
                dig_dh,
                background=[(d["onset"], d["bottom"]) for d in h_drops],
            ),
            g(
                "radius_turn",
                "radius",
                radius,
                "radius: after the big dip and upswing, where the downward trend begins",
                dig_radius,
                background=[(d["start"], d["end"]) for d in radius_dips],
            ),
            g(
                "x_drop_end",
                "bucket_x",
                x,
                "bucket x (toward truck +): arrived back at the pile side after the big drop",
                dig_x,
                mode="fixed",
                background=[(d["onset"], d["bottom"]) for d in x_drops],
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
                role="required",
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
            g(
                "radius_dip_start",
                "radius",
                radius,
                "radius: the start of a dip -- a knee from flat to falling",
                haul_radius,
                background=[(d["start"], d["end"]) for d in radius_dips],
            ),
        ],
        "dumping": [
            g(
                "aspect_steepest",
                "aspect_ratio",
                F["aspect_ratio"],
                "box aspect ratio: the steepest part of a drop -- every drop of >= 30% "
                "is a candidate; the one the other cues support most is kept",
                aspect_steepest,
                background=[(d["onset"], d["bottom"]) for d in aspect_candidates],
                role="primary",
                candidates=aspect_all,
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
                "height: the middle of the climb into the second hump (the hump's "
                "maximum must exist, but is not in the window)",
                height_second_hump,
                before_end=True,
                background=[(float(t[p]) - 0.1, float(t[p]) + 0.1) for p in h_peaks],
            ),
            check(
                "dh_near_0",
                "dh_dt",
                dh,
                "dh/dt: close to or at 0 (within 3x its noise)",
                dump_dh_mask,
            ),
            check(
                "dx_positive", "dx_dt", dx, "dx/dt (toward truck +): positive", dump_dx_mask
            ),
            check(
                "overlapping",
                "truck_overlap",
                overlap,
                "bucket ^ truck box: overlapping the truck",
                dump_overlap_mask,
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
                "height_peak",
                "height",
                h,
                "height: last top before the big drop, where the climb gets within 20% of it",
                swing_height,
                background=h_high_regions,
            ),
            g(
                "dh_positive_falling",
                "dh_dt",
                dh,
                "dh/dt: positive and decreasing",
                first_run(swing_dh_mask, "mid"),
                background=cc._runs(swing_dh_mask, t),
            ),
            g(
                "dx_bump",
                "dx_dt",
                dx,
                "dx/dt (toward truck +): where the upward bump starts, before the big dip",
                swing_dx_bump,
                background=[(d["start"], d["end"]) for d in dx_dips],
            ),
            g(
                "radius_bump_start",
                "radius",
                radius,
                "radius: where the steep part of the big bump begins",
                swing_radius,
                background=[(d["start"], d["end"]) for d in radius_peaks],
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
            & (side_signals(F, side)[0][2] < 0)
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
    hi = c + half
    if graph.get("before_end"):  # e.g. dump height: the hump's maximum must not be inside
        hi = min(hi, b - 1e-3)
    return {
        **found,
        "half": half,
        "raw": raw,
        "capped": capped,
        "window": (max(c - half, anchor), hi),
    }


def _runs_in(t, mask, lo, hi):
    return [(a, b) for a, b in cc._runs(mask, t) if b >= lo and a <= hi]


def dig_reference(t, F, anchor, cycle):
    """This cycle's dig height: the median height inside the dig window (the running
    median since the search start when no dig window was found)."""
    w = cycle.get("dig_window")
    if w is not None:
        inside = (t >= w[0]) & (t <= w[1])
        return np.full(len(t), float(np.nanmedian(F["height"][inside])))
    return cc._running_median(F["height"], int(np.searchsorted(t, anchor)))


def side_signals(F, side):
    """The two "which side" signals the gates read: the bucket's side of the TRUCK
    (< 0 = pile side) and of the CABIN (> 0 = truck side). With the default pile-truck
    line both come from positions along that line, so no image direction is assumed;
    otherwise from image x, signed toward the truck."""
    if HORIZONTAL["mode"] == "pile_truck":
        pos = F["pile_truck_pos"]
        return (
            ("pile_truck_pos", "pile-to-truck position < 0", pos),
            (
                "pos_past_cabin",
                "pile-to-truck position past the cabin's",
                pos - F["cabin_pile_truck_pos"],
            ),
        )
    return (
        ("rel_truck_x", "bucket - truck x < 0", F["rel_truck_x"] * side),
        ("rel_cabin_x", "bucket - cabin x > 0", F["rel_cabin_x"] * side),
    )


def gates(t, F, side, phase, anchor, cycle):
    """STEP 1 -- the necessary thresholds. None of a phase's cues are considered
    until all of its gates hold (Aanya: "some of the things are above a certain
    threshold that's necessary, so none of the cues are even considered until that
    happens"). Each is (id, label, signal key, mask)."""
    h, overlap = F["height"], F["truck_overlap"]
    (tk, tlabel, tgap), (ck, clabel, cgap) = side_signals(F, side)
    with np.errstate(invalid="ignore"):
        if phase == "digging":
            return [
                ("D1", "no overlap with the truck box", "truck_overlap", overlap <= 0),
                (
                    "D2",
                    f"on the pile side of the truck ({tlabel})",
                    tk,
                    tgap < 0,
                ),
            ]
        if phase == "hauling":
            return [
                (
                    "H1",
                    "height above this cycle's dig height",
                    "height",
                    h > dig_reference(t, F, anchor, cycle),
                ),
                ("H0", "no overlap with the truck box", "truck_overlap", overlap <= 0),
            ]
        if phase == "dumping":
            return [
                (
                    "P1",
                    f"on the truck side of the cabin ({clabel})",
                    ck,
                    cgap > 0,
                ),
                (
                    "P2",
                    "height above this cycle's dig height",
                    "height",
                    h > dig_reference(t, F, anchor, cycle),
                ),
                (
                    "P3",
                    "above the cabin (bucket - cabin y > 0)",
                    "rel_cabin_y",
                    F["rel_cabin_y"] > 0,
                ),
            ]
    return []


def fill_checks(t, graphs, windows, primary):
    """Dump's "check" cues: a condition read inside the primary (aspect) window."""
    for k, g in enumerate(graphs):
        if g["mode"] != "check":
            continue
        windows[k] = None
        if primary is None:
            continue
        a, b = primary["window"]
        if (
            DUMP_REWEIGHT["on"]
            and g["id"] == "overlapping"
            and primary.get("centre") is not None
        ):
            at = int(np.argmin(np.abs(np.asarray(t) - primary["centre"])))
            if not g["mask"][at]:  # read at the tip itself
                continue
        inside = [(max(x, a), min(y, b)) for x, y in _runs_in(t, g["mask"], a, b)]
        if inside:
            windows[k] = {
                "event": (inside[0][0], inside[-1][1]),
                "centre": None,
                "context": None,
                "half": None,
                "raw": None,
                "capped": False,
                "window": (inside[0][0], inside[-1][1]),
            }


def coverage(t, windows):
    count = np.zeros(len(t), int)
    for w in windows:
        if w is not None:
            count += (t >= w["window"][0] - 1e-9) & (t <= w["window"][1] + 1e-9)
    return count


# Tier weights -- strong 3, medium 2, weak 1 -- agreed with Aanya on 2026-09-27
# ("some of these are a lot more valuable cues than others"). Set by judgment about
# which motions really mark a phase, informed by (not fitted to) how often each cue
# contained the marks: fitting them to these marks would only learn these clips.
# Cues that only showed up on THESE clips' graphs, with no physical reason they must
# happen on another video (the universality audit). Aanya: "let's get rid of the only
# seen on graphs (radius) cues". Their finders stay, so a new clip can confirm one and
# bring it back; they don't vote.
# In the pile-truck mode the horizontal cues read the pile-to-truck line: the signal
# key and the label's prefix each cue shows
PILE_TRUCK_NAMES = {
    "bucket_x": ("pile_truck_pos", ("bucket x (toward truck +)", "pile-to-truck position")),
    "rel_truck_x": (
        "pile_truck_pos",
        ("bucket - truck x (toward truck +)", "pile-to-truck position"),
    ),
    "dx_dt": ("pile_truck_pos_dt", ("dx/dt (toward truck +)", "pile-to-truck speed")),
}
HORIZONTAL = {"mode": "pile_truck"}  # x | distance | truck_x | pile_truck; set by main

SEEN_ONLY = {
    "radius_turn",
    "radius_dip_start",
    "radius_bump_start",
    "dx_min",
    "dx_bump",
    "cabin_x_going_up",
}

WEIGHTS = {
    ("digging", "speed_min"): 3,
    ("digging", "height_drop_end"): 3,
    ("digging", "x_drop_end"): 3,  # the bucket back at the pile side: distance to the truck
    ("digging", "dx_crossing"): 2,
    ("digging", "radius_turn"): 2,
    ("digging", "dh_back_to_0"): 1,
    ("hauling", "height_takeoff"): 3,
    ("hauling", "truck_x_takeoff"): 3,
    ("hauling", "radius_dip_start"): 2,
    ("hauling", "dx_min"): 1,
    ("dumping", "cabin_x_going_up"): 2,
    ("dumping", "height_second_hump"): 2,
    ("dumping", "dh_near_0"): 1,
    ("dumping", "dx_positive"): 1,
    ("dumping", "overlapping"): 1,
    ("swinging", "x_takeoff"): 3,
    ("swinging", "radius_bump_start"): 2,
    ("swinging", "dh_positive_falling"): 1,
    ("swinging", "height_peak"): 1,
    ("swinging", "dx_bump"): 1,
}


def combine(t, phase, graphs, windows, cue_from):
    """STEPS 2 and 3, WEIGHTED: a window needs more than half of the total weight of
    the cues that found something. Returns (window, weight agreeing, weight needed,
    core) -- the core being where they overlap."""
    roles = [g["role"] for g in graphs]
    wt = [WEIGHTS.get((phase, g["id"]), 1) for g in graphs]
    if phase == "dumping":  # the aspect ratio's window, if most of the rest's weight agrees
        k0 = roles.index("primary")
        primary = windows[k0]
        if primary is None:
            return None, 0, 0, None
        a, b = primary["window"]
        support = [(w, wt[k]) for k, w in enumerate(windows) if k != k0 and w is not None]
        # EXPERIMENT: the majority is out of ALL the supporting cues' weight, not
        # only the cues that found something -- one cue alone is not a majority
        total = sum(x for k, x in enumerate(wt) if k != k0)
        agree = sum(x for w, x in support if w["window"][0] <= b and w["window"][1] >= a)
        return (a, b), agree, total / 2, (a, b)
    if phase == "hauling":  # height must be one of them
        req = windows[roles.index("required")]
        if req is None:
            return None, 0, 0, None
        # A supporting cue whose event is far from height's window found something
        # else -- not evidence about this haul: not counted.
        ra, rb = req["window"]
        windows = [
            None
            if w is None or w["window"][0] > rb + NEAR or w["window"][1] < ra - NEAR
            else w
            for w in windows
        ]
    present = [(w, wt[k]) for k, w in enumerate(windows) if w is not None]
    # EXPERIMENT: out of ALL the stage's cue weight, not only the cues that found
    # something (a lone weak cue used to count as a unanimous vote)
    total = sum(wt)
    count = np.zeros(len(t))
    for w, x in present:
        count += x * ((t >= w["window"][0] - 1e-9) & (t <= w["window"][1] + 1e-9))
    ok = (2 * count > total) & (t >= cue_from - 1e-9)
    if phase == "hauling":
        ok &= (t >= ra - 1e-9) & (t <= rb + 1e-9)
    runs = cc._runs(ok, t)
    if not runs:
        # "there has to be something for that given stage transition ... have the bar
        # for triggering the stage lower, but we just rely on these cues to try to
        # centralize where it happened": no location has more than half the weight,
        # so take where the MOST weight agrees -- inside height's window for haul --
        # and flag it as low agreement rather than dropping the phase
        allowed = t >= cue_from - 1e-9
        if phase == "hauling":
            allowed &= (t >= ra - 1e-9) & (t <= rb + 1e-9)
        if not allowed.any() or count[allowed].max() <= 0:
            return None, 0, total / 2, None
        best = count[allowed].max()
        runs = cc._runs(allowed & (count >= best - 1e-9), t)
    a, b = runs[0]
    n = float(count[(t >= a) & (t <= b)].max())
    # Once most of the weight agrees, the window is the FULL extent of the cues that
    # agree -- "if it's validated across three cues, then wider is better" -- capped
    # at 2 x MAX_HALF around the middle of their overlap.
    agreeing = [w for w, _ in present if w["window"][0] <= b and w["window"][1] >= a]
    lo = max(min(w["window"][0] for w in agreeing), cue_from)
    hi = max(w["window"][1] for w in agreeing)
    mid = (a + b) / 2
    return (max(lo, mid - MAX_HALF), min(hi, mid + MAX_HALF)), n, total / 2, (a, b)


def step(t, F, side, found, phase, anchor, cycle, clip_start=True):
    if clip_start and phase == "digging" and anchor <= t[0] + 1e-9:
        c = clip_start_dig(t, F, side)
        if c is not None:
            window = (max(c - MIN_HALF, t[0]), c + MIN_HALF)
            return {
                "phase": phase,
                "search_from": anchor,
                "graphs": [None] * len(found[phase]),
                "window": window,
                "votes": 0,
                "need": 0,
                "agreed": True,
                "start": window[0],
                "clip_start": c,
                # the clip-start rule includes the dig gates (no overlap, pile side):
                # recorded over its first seconds so the page shows where they hold
                "gates": [
                    {
                        "id": gid,
                        "label": lab,
                        "key": key,
                        "runs": _runs_in(t, m, anchor, anchor + CLIP_START_SECONDS),
                    }
                    for gid, lab, key, m in gates(t, F, side, phase, anchor, cycle)
                ],
                "gates_open": c,
                "cue_from": anchor,
            }
    graphs = found[phase]
    cycle["search_from"] = anchor
    gate_list = gates(t, F, side, phase, anchor, cycle)
    gate = np.ones(len(t), bool)
    for *_, m in gate_list:
        gate &= m
    open_idx = next((j for j in range(_i(t, anchor), len(t)) if gate[j]), None)
    gate_info = [
        {"id": gid, "label": lab, "key": key, "runs": _runs_in(t, m, anchor, anchor + 40)}
        for gid, lab, key, m in gate_list
    ]
    base = {"phase": phase, "search_from": anchor, "gates": gate_info}
    if open_idx is None:
        return {
            **base,
            "graphs": [None] * len(graphs),
            "window": None,
            "votes": 0,
            "need": 0,
            "agreed": False,
            "start": anchor,
            "gates_open": None,
            "cue_from": None,
        }
    opened = float(t[open_idx])
    cue_from = max(anchor, opened - GATE_LOOKAHEAD)

    def passes(event):
        a, b = event
        return bool(gate[_i(t, a) : _i(t, b + GATE_LOOKAHEAD) + 1].any())

    windows = [None] * len(graphs)
    for k, g in enumerate(graphs):
        if g["mode"] == "check":
            continue
        e, frm = g["find"](cue_from, cycle), cue_from
        for _ in range(30):  # skip events that fail the gates
            if e is None or passes(e["event"]):
                break
            frm = e["event"][1] + 1e-3
            e = g["find"](frm, cycle)
        windows[k] = None if e is None else widen(t, g, e, cue_from)
    if phase == "dumping":
        k0 = [g["role"] for g in graphs].index("primary")
        pg = graphs[k0]
        cands = [e for e in pg["candidates"](cue_from, cycle) if passes(e["event"])]
        # The truck visit brackets the dump: a bucket can only dump while over the
        # truck, and tips from a LATER visit belong to a later cycle -- without
        # this, a clean dump one cycle on outscored this cycle's real tip. Only
        # tips inside the first visit that ends after the search start compete.
        visit = next((v for v in cc._runs(F["truck_overlap"] > 0, t) if v[1] > cue_from), None)
        if visit is not None:
            cands = [e for e in cands if visit[0] - 1e-9 <= e["centre"] <= visit[1] + 1e-9]
        tried = []
        for e in cands:  # each candidate tip, with the checks read inside ITS window
            trial = list(windows)
            trial[k0] = widen(t, pg, e, cue_from)
            fill_checks(t, graphs, trial, trial[k0])
            got = combine(t, phase, graphs, trial, cue_from)
            tried.append((got[1], -e["centre"], trial, got, e["centre"]))
        candidates = [{"centre": c, "support": n} for n, _, _, _, c in tried]
        if tried:  # the most support wins; ties go to the earlier candidate
            best = max(tried, key=lambda x: (x[0], x[1]))
            windows = best[2]
            window, n, need, core = best[3]
        else:
            window, n, need, core = None, 0, 0, None
    else:
        candidates = None
        window, n, need, core = combine(t, phase, graphs, windows, cue_from)
    agreed = window is not None
    weak = agreed and not (2 * n > 2 * need if need else True)
    return {
        **base,
        "candidates": candidates,
        "weak": bool(weak),
        "graphs": windows,
        "window": window,
        "core": core,
        "votes": n,
        "need": need,
        "agreed": agreed,
        "start": window[0] if agreed else cue_from,
        "gates_open": opened,
        "cue_from": cue_from,
    }


def from_marks(t, F, side, marks):
    """Each marked onset, searched for after the previous MARKED onset."""
    found = {
        ph: [g for g in lst if g["id"] not in SEEN_ONLY]
        for ph, lst in finders(t, F, side).items()
    }
    out, cycle = [], {}
    for k, (phase, _) in enumerate(marks):
        if phase == "digging":
            cycle = {}
        s = step(t, F, side, found, phase, marks[k - 1][1] if k else float(t[0]), cycle)
        if s is not None:
            if phase == "digging":
                # later rules read "the dig window" -- the haul height line, the dig
                # height gates -- from the CORE where the dig cues agree, not the
                # widened output, which reaches back into the swing's descent
                cycle["dig_window"] = s.get("core") or s["window"]
            out.append(s)
    return found, out


PHASE_ORDER = ("digging", "hauling", "dumping", "swinging")


def dump_stretches(t, F):
    """Where the bucket is over the truck long enough to dump: overlap stretches at
    least half as long as the longest one in the video. The short passes over the
    truck at the start of each return swing (about 1 s here) do not count."""
    runs = cc._runs(F["truck_overlap"] > 0, t)
    if not runs:
        return []
    longest = max(b - a for a, b in runs)
    return [(a, b) for a, b in runs if b - a >= 0.5 * longest]


def open_stretches(t, F):
    """The stretches between dump stretches: where dig, haul and the swing happen."""
    edges = [float(t[0])] + [x for d in dump_stretches(t, F) for x in d] + [float(t[-1])]
    return [(a, b) for a, b in zip(edges[0::2], edges[1::2], strict=True) if b > a]


def climb_start(t, h, a, b, last):
    """Where the lift to the truck leaves pile level, in the open stretch (a, b): the
    last time height rises through HALFWAY between the stretch's pile level (its 5th
    percentile) and its height on arriving over the truck, walked back to where it was
    still within a tenth of that climb of pile level. A dig must end before this:
    "how do we make sure that it doesn't accidentally encompass haul?" The last
    stretch has no truck after it, so no bound."""
    if last:
        return b
    m = (t >= a) & (t <= b)
    lo, hi = pile_and_arrival(t, h, a, b, last)
    if hi <= lo:
        return b
    idx = np.flatnonzero(m)
    ups = [i for i in idx[1:] if h[i - 1] < (lo + hi) / 2 <= h[i]]
    if not ups:
        return b
    k = ups[-1]
    while k > idx[0] and h[k] > lo + 0.1 * (hi - lo):
        k -= 1
    return float(t[k])


def pile_and_arrival(t, h, a, b, last):
    """The open stretch's pile level (5th percentile of its height) and its height on
    arriving over the truck at its end (95th percentile for the last stretch, which
    has no truck after it)."""
    m = (t >= a) & (t <= b)
    lo = float(np.nanpercentile(h[m], 5))
    hi = float(np.nanpercentile(h[m], 95)) if last else float(h[max(_i(t, b) - 1, 0)])
    return lo, hi


def settled_in_pile(t, F, side, a, b, levels):
    """WEAK dig signal, used only when the dig cues find nothing in the stretch: the
    first moment in (a, b) the bucket is low (within a quarter of the way from the
    stretch's pile level to its arrival height), on the pile side, off the truck and
    at rest (2D speed within 3x its noise, at least 2% of its range)."""
    m = (t >= a) & (t <= b)
    if not m.any():
        return None, []
    h, speed = F["height"], F["speed_2d"]
    p5, p95 = levels
    rest = max(3 * noise_scale(speed), 0.02 * float(np.nanmax(speed) - np.nanmin(speed)))
    low = p5 + 0.25 * (p95 - p5)
    with np.errstate(invalid="ignore"):
        conditions = [
            (
                "low",
                f"height <= {low:+.2f} (a quarter of the way up to the truck)",
                "height",
                h <= low,
            ),
            (
                "pile side",
                "pile-to-truck position < 0",
                "pile_truck_pos",
                side_signals(F, side)[0][2] < 0,
            ),
            (
                "off the truck",
                "no overlap with the truck box",
                "truck_overlap",
                F["truck_overlap"] <= 0,
            ),
            ("at rest", f"2D speed <= {rest:.3f} (its rest band)", "speed_2d", speed <= rest),
        ]
    ok = m.copy()
    for *_, c in conditions:
        ok &= c
    drawn = [
        {"id": cid, "label": lab, "key": key, "runs": _runs_in(t, c & m, a, b)}
        for cid, lab, key, c in conditions
    ]
    idx = np.flatnonzero(ok)
    return (None if idx.size == 0 else float(t[idx[0]])), drawn


def find_dig(t, F, side, found, anchor, cycle):
    """A dig, in the open stretch the search is in -- or, if nothing there, the next
    one (Aanya: "if you can't find a cue for digging in that then look at the next one
    and it has to be within one of those two"). In each stretch: the dig cues first,
    then the weak "settled in the pile" signal, before moving on -- so a clip that
    opens mid-dig, with no return swing before it for the cues to read, keeps its
    first dig. Every dig ends before the stretch's climb to the truck."""
    stretches = open_stretches(t, F)
    near = [k for k, (a, b) in enumerate(stretches) if b > anchor + 1e-6][:2]
    for k in near:
        a, b = stretches[k]
        lo = max(a, anchor)
        hi = climb_start(t, F["height"], a, b, last=k == len(stretches) - 1)
        if hi <= lo:
            continue
        s = step(t, F, side, found, "digging", lo, cycle, clip_start=False)
        if s is not None and s["window"] is not None:
            ca, cb = s.get("core") or s["window"]
            if lo - 1e-9 <= (ca + cb) / 2 <= hi:
                w = (s["window"][0], min(s["window"][1], hi))
                return {**s, "window": w, "how": "dig cues", "region": (lo, hi)}
        levels = pile_and_arrival(t, F["height"], a, b, last=k == len(stretches) - 1)
        c, conditions = settled_in_pile(t, F, side, lo, hi, levels)
        if c is not None:
            w = (max(c - MIN_HALF, lo), min(c + MIN_HALF, hi))
            base = s or step(t, F, side, found, "digging", lo, cycle, clip_start=False)
            return {
                **base,
                "window": w,
                "core": w,
                "weak": True,
                "clip_start": c,
                "how": "settled in the pile (weak: no dig cue in this stretch)",
                "weak_signal": {"at": c, "conditions": conditions},
                "region": (lo, hi),
            }
    return None


def from_found(t, F, side):
    """NO LABELS: each stage is searched for after the END of the window this code
    found for the stage before it -- dig, haul, dump, swing, round again -- from the
    clip's first frame (Aanya: "we need to make sure the condition of the end of the
    prev interval holds before looking for the next stage"). A stage is never
    skipped: if one gets no window, the search stops there and says so. Digs are
    found by find_dig (inside an open stretch, before its climb to the truck)."""
    found = {
        ph: [g for g in lst if g["id"] not in SEEN_ONLY]
        for ph, lst in finders(t, F, side).items()
    }
    out, cycle, anchor, k, stop = [], {}, float(t[0]), 0, None
    while anchor < t[-1]:
        phase = PHASE_ORDER[k % len(PHASE_ORDER)]
        if phase == "digging":
            cycle = {}
            s = find_dig(t, F, side, found, anchor, cycle)
        else:
            s = step(t, F, side, found, phase, anchor, cycle)
        if s is None or s["window"] is None:
            stop = {"phase": phase, "search_from": anchor, "why": "no window found"}
            break
        if phase == "digging":
            cycle["dig_window"] = s.get("core") or s["window"]
        out.append(s)
        nxt = s["window"][1]
        if nxt <= anchor + 1e-6:
            stop = {"phase": phase, "search_from": anchor, "why": "did not move forward"}
            break
        anchor, k = nxt, k + 1
    return found, out, stop


def score(steps, marks):
    """Marks are used ONLY here: for each labelled onset, the found window of the same
    stage that contains it, or else the nearest one."""
    rows = []
    for phase, m in marks:
        same = [s for s in steps if s["phase"] == phase and s["window"]]
        if not same:
            rows.append({"phase": phase, "mark": m, "inside": False, "gap": None})
            continue
        best = min(same, key=lambda s: max(s["window"][0] - m, m - s["window"][1], 0.0))
        gap = max(best["window"][0] - m, m - best["window"][1], 0.0)
        rows.append(
            {
                "phase": phase,
                "mark": m,
                "inside": gap == 0.0,
                "gap": gap,
                "window": best["window"],
            }
        )
    used = {tuple(r["window"]) for r in rows if r.get("window")}
    extra = [s["window"] for s in steps if s["window"] and tuple(s["window"]) not in used]
    return {"rows": rows, "extra_windows": extra}


def _clean(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", choices=sorted(cc.CLIPS), default="long")
    ap.add_argument(
        "--features", type=Path, help="any pipeline features.npz (instead of --clip)"
    )
    ap.add_argument(
        "--labels", type=Path, help="labels for --features, used only to score (optional)"
    )
    ap.add_argument(
        "--anchor",
        choices=("found", "marks"),
        default="found",
        help="start each stage's search after the window found for the stage before "
        "(default: runs with no labels) or after the labelled mark (testing only)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--horizontal",
        choices=("pile_truck", "x", "distance", "truck_x"),
        default="pile_truck",
        help="horizontal cues from the position along the pile-to-truck line (default), "
        "bucket x, the distance to the truck, or the x gap to it",
    )
    ap.add_argument(
        "--dump-reweight",
        action="store_true",
        help="EXPERIMENT: over the truck votes 3 for a dump (at the tip, first 70%% of "
        "the visit), dh/dt near 0 votes 0.5",
    )
    args = ap.parse_args(argv)
    HORIZONTAL["mode"] = args.horizontal
    if args.dump_reweight:
        DUMP_REWEIGHT["on"] = True
        WEIGHTS[("dumping", "overlapping")] = 3
        WEIGHTS[("dumping", "dh_near_0")] = 0.5
    if args.features is not None:
        feat_path, label_path = args.features, args.labels
    else:
        feat_path, label_path = cc.CLIPS[args.clip]
    t, F = cc.load_features(feat_path)
    side = cc.truck_side(F)
    marks = cc.load_onsets(label_path) if label_path is not None else []
    stop = None
    if args.anchor == "marks":
        if not marks:
            raise SystemExit("--anchor marks needs labels")
        found, steps = from_marks(t, F, side, marks)
    else:
        found, steps, stop = from_found(t, F, side)

    data = {
        "clip": args.features.parent.name if args.features is not None else args.clip,
        "side": side,
        "anchor": args.anchor,
        "stop": stop,
        "score": score(steps, marks) if marks else None,
        "limits": {
            "min_half": MIN_HALF,
            "rate_min_half": RATE_MIN_HALF,
            "max_half": MAX_HALF,
            "slope_side": SLOPE_SIDE,
        },
        "times": [round(float(v), 3) for v in t],
        "signals": {
            **{
                g["key"]: [_clean(v) for v in g["values"]] for ph in found.values() for g in ph
            },
            # the gates' signals, signed like the cues' (horizontal: + toward the truck)
            "rel_truck_x": [_clean(v) for v in F["rel_truck_x"] * side],
            "rel_cabin_x": [_clean(v) for v in F["rel_cabin_x"] * side],
            "rel_cabin_y": [_clean(v) for v in F["rel_cabin_y"]],
            "truck_overlap": [_clean(v) for v in F["truck_overlap"]],
            "pile_truck_pos": [_clean(v) for v in F["pile_truck_pos"]],
            "pos_past_cabin": [
                _clean(v) for v in F["pile_truck_pos"] - F["cabin_pile_truck_pos"]
            ],
            # x vs straight-line distance to the truck, drawn side by side
            "truck_distance": [_clean(v) for v in F["truck_distance"]],
            "toward_dist_dt": [_clean(v) for v in -F["truck_distance_dt"]],
            "toward_dx_dt": [_clean(v) for v in F["dx_dt"] * side],
        },
        "horizontal_mode": HORIZONTAL["mode"],
        "pile_truck_angle": float(F["pile_truck_angle"][0]),
        # the big dips the dig "dx crosses back to 0" cue picks from, per signal
        "horizontal_dips": {
            "toward_dx_dt": [
                d["apex"]
                for d in cc.excursion_windows(F["dx_dt"] * side, t, "dip", min_size=0.5)
            ],
            "toward_dist_dt": [
                d["apex"]
                for d in cc.excursion_windows(-F["truck_distance_dt"], t, "dip", min_size=0.5)
            ],
        },
        "finders": {
            ph: [
                {
                    "id": g["id"],
                    "signal": g["key"],
                    "label": g["label"],
                    "mode": g["mode"],
                    "spans": g["background"],
                    "role": g["role"],
                    "weight": WEIGHTS.get((ph, g["id"]), 1),
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
    if stop:
        print(
            f"  stopped at {stop['phase']} (searched from {stop['search_from']:.2f}s): "
            f"{stop['why']}"
        )
    if marks:
        sc = data["score"] if data["score"] else score(steps, marks)
        n_in = sum(r["inside"] for r in sc["rows"])
        print(f"  labels (scoring only): {n_in}/{len(sc['rows'])} marks inside a window")
        for r in sc["rows"]:
            if not r["inside"]:
                gap = "no window" if r["gap"] is None else f"{r['gap']:.2f} s outside"
                print(f"    missed {r['phase']} at {r['mark']:.2f}s: {gap}")
        if sc["extra_windows"]:
            print(f"  windows with no mark: {len(sc['extra_windows'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
