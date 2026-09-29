"""Pass 1 of the phase search: find the interval each phase starts in.

Each phase is searched for after the previous phase's window (the first dig: from the
clip's start, with the clip-start rule), so the order dig -> haul -> dump -> swing is
enforced. For each phase, in this order (Aanya, 2026-09-27):

1. GATES -- the necessary thresholds. None of the phase's cues are considered until
   all of its gates hold; cues may begin ``GATE_LOOKAHEAD`` s before that, and an
   event only counts if the gates hold somewhere in it or the second after it.
   Dig: not over the truck, on the pile side of it. Haul: height above this
   cycle's dig height, not over the truck. Dump: on the truck side of the cabin,
   height above the dig height. Return swing: none.
2. REQUIRED CUE -- haul: height must be part of the window. Dump: the aspect
   ratio's steepest drop IS the window.
3. SUPPORTING CUES -- a majority of the cue weight must agree (overlap the window;
   for dump, a majority of the rest must overlap the aspect window). Every cue is a
   pattern from Aanya's key, skipping graphs that are only a constant shift of another.

A cue's window is its event's centre widened by its timing uncertainty (noise /
slope change), clamped to ``MIN_HALF``..``MAX_HALF``; a span cue is its span padded
by the minimum; a "check" cue (dump) is a condition read inside the aspect window.

Nothing here reads a label: labels only score the result, outside ``src/``.
`starts.py` picks one frame inside each window.
"""

from __future__ import annotations

import numpy as np

from . import cues
from .rates import noise_scale

# Window half-widths, in seconds. Minimum +/-0.75 s and maximum +/-1.2 s -- Aanya,
# 2026-09-27; broadened (wider is safer once several cues validate a window --
# missing the transition cannot be undone). Rate signals are smoothed over 0.9 s, so
# their events cannot be placed more tightly than about +/-0.45 s. The maximum stays
# under the shortest phase (a dump, about 2.5 s), so no window can hold two phase
# starts.
MIN_HALF = 0.75
RATE_MIN_HALF = 0.75
MAX_HALF = 1.2
SLOPE_SIDE = 1.0  # seconds either side of an event for the slope-change estimate
RATES = {"dh_dt", "speed_2d", "pile_truck_pos_dt"}
GATE_LOOKAHEAD = 1.0  # s after a haul event in which its gates may be met
HAUL_HEIGHT_SECONDS = 1.5  # "a second long or something", broadened from 1.0
NEAR = 5.0  # s: a haul cue further than this from height's window is not counted
STEEP = 0.25  # a 2 s line moving >= this fraction of the spread is a steep rise


# ------------------------------------------------------------------ uncertainty


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
    rel_cabin_x = F["rel_cabin_x"] * side
    # "the goal is to get rid of the position specific ones": where the bucket is
    # along the line from the pile to the truck (-1 pile, 0 truck, + past it), the
    # line's direction found from the video, not assumed to be image left-right
    x = F["pile_truck_pos"]
    dx = F["pile_truck_pos_dt"]
    rel_truck_x = x
    overlap = F["truck_overlap"]

    humps = cues.excursion_windows(speed, t, "peak", min_size=0.5)
    dx_dips = cues.excursion_windows(dx, t, "dip", min_size=0.5)
    h_drops = cues.big_drops(h, t, min_drop=0.5, max_seconds=6.0)
    aspect_drops = cues.big_drops(F["aspect_ratio"], t)
    # "You can detect them multiple times, and then you can figure out which is the
    # real signal based on validating across multiple": every drop of >= 30% of the
    # range is a candidate tip; step() keeps the one the other cues support most
    aspect_candidates = cues.big_drops(F["aspect_ratio"], t, min_drop=0.3)

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
    # "consider more of the knee" (Aanya). The STRICT knee picks WHICH bend (the sharp
    # one); a LOOSER knee -- before moves < 2x the flat limit, after rises >= half the
    # steep limit -- measures how far that bend extends. The loose one alone also
    # matched a smaller bend mid-dig.
    takeoff_tx = cues.knee_mask(rel_truck_x, t, "rising", "flat_to_steep") & (rel_truck_x < 0)
    bend_tx = cues.runs(
        cues.knee_mask(
            rel_truck_x,
            t,
            "rising",
            "flat_to_steep",
            flat_max=2 * cues.FLAT,
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
        for a, b in cues.runs(mask, t):
            if b < anchor:
                continue
            a = max(a, anchor)
            if gate[_i(t, a) : _i(t, b + GATE_LOOKAHEAD) + 1].any():
                return (a, b)
        return None

    _, h_after = cues.side_changes(h, t, cues.SIDE)

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
                if h[j] > level and np.isfinite(h_after[j]) and h_after[j] >= cues.FLAT
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
    _, cabin_after = cues.side_changes(rel_cabin_x, t, cues.SIDE)
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
    dh = F["dh_dt"]
    dh_noise = noise_scale(dh)
    x_drops = cues.big_drops(x, t, min_drop=0.5, max_seconds=6.0)
    x_noise = noise_scale(x)

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

    with np.errstate(invalid="ignore"):
        dump_dh_mask = np.abs(dh) <= 3 * dh_noise  # key: "close to or at 0"
        dump_dx_mask = dx > 0  # key: "positive"
        dump_overlap_mask = overlap > 0  # key (bucket - truck x): "overlap"
        swing_dh_mask = (dh > dh_noise) & (np.gradient(dh, t) < 0)  # "positive decreasing"

    def first_run(mask, at="start"):
        def find(anchor, _cycle):
            e = first_event(cues.runs(mask, t), anchor)
            if e is None:
                return None
            return {
                "event": e,
                "centre": e[0] if at == "start" else sum(e) / 2,
                "context": None,
            }

        return find

    # ---- swing
    # either way: on the long clip the bucket first moves past the truck, on Untitled4
    # it heads straight back toward the pile (the "swing overshoot" question)
    x_takeoff = sorted(
        cues.knee_windows(x, t, "rising", "flat_to_steep")
        + cues.knee_windows(x, t, "falling", "flat_to_steep")
    )
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
        find,
        mode="centre",
        min_half=None,
        role="support",
        mask=None,
        candidates=None,
        before_end=False,
    ):
        return {
            "id": id_,
            "key": key,  # the signal the cue reads; decides which minimum half-width applies
            "values": values,
            "find": find,
            "mode": mode,
            "min_half": min_half,
            "role": role,  # "required" (haul), "primary" (dump) or "support"
            "mask": mask,  # mode "check": a condition checked inside the primary window
            "candidates": candidates,  # primary only: every candidate event, to validate
            "before_end": before_end,  # the window must stop before the event's end
        }

    def check(id_, key, values, mask):
        return g(id_, key, values, None, mode="check", mask=mask)

    return {
        "digging": [
            g("speed_min", "speed_2d", speed, speed_min),
            g("height_drop_end", "height", h, height_drop_end),
            g("dx_crossing", "pile_truck_pos_dt", dx, dx_crossing, mode="span"),
            g("dh_back_to_0", "dh_dt", dh, dig_dh),
            g("x_drop_end", "pile_truck_pos", x, dig_x, mode="fixed"),
        ],
        "hauling": [
            g("height_takeoff", "height", h, haul_height, mode="fixed", role="required"),
            g(
                "truck_x_takeoff",
                "pile_truck_pos",
                rel_truck_x,
                haul_truck_x,
                mode="span",
                min_half=0.5,
            ),
        ],
        "dumping": [
            g(
                "aspect_steepest",
                "aspect_ratio",
                F["aspect_ratio"],
                aspect_steepest,
                role="primary",
                candidates=aspect_all,
            ),
            g("height_second_hump", "height", h, height_second_hump, before_end=True),
            check("dh_near_0", "dh_dt", dh, dump_dh_mask),
            check("dx_positive", "pile_truck_pos_dt", dx, dump_dx_mask),
            check("overlapping", "truck_overlap", overlap, dump_overlap_mask),
        ],
        "swinging": [
            g("x_takeoff", "pile_truck_pos", x, from_spans(x_takeoff)),
            g("height_peak", "height", h, swing_height),
            g("dh_positive_falling", "dh_dt", dh, first_run(swing_dh_mask, "mid")),
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
        ref = cues.running_median(F["height"], int(np.searchsorted(t, anchor)))
    with np.errstate(invalid="ignore"):
        return (F["height"] > ref) & (F["truck_overlap"] <= 0)


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
    return [(a, b) for a, b in cues.runs(mask, t) if b >= lo and a <= hi]


def dig_reference(t, F, anchor, cycle):
    """This cycle's dig height: the median height inside the dig window (the running
    median since the search start when no dig window was found)."""
    w = cycle.get("dig_window")
    if w is not None:
        inside = (t >= w[0]) & (t <= w[1])
        return np.full(len(t), float(np.nanmedian(F["height"][inside])))
    return cues.running_median(F["height"], int(np.searchsorted(t, anchor)))


def side_signals(F):
    """The two "which side" signals the gates read, both positions along the pile-to-truck
    line so no image direction is assumed: the bucket's side of the TRUCK (< 0 = pile
    side) and of the CABIN (> 0 = truck side)."""
    pos = F["pile_truck_pos"]
    return pos, pos - F["cabin_pile_truck_pos"]


def gates(t, F, phase, anchor, cycle):
    """STEP 1 -- the necessary thresholds. None of a phase's cues are considered
    until all of its gates hold (Aanya: "some of the things are above a certain
    threshold that's necessary, so none of the cues are even considered until that
    happens"). Returns one boolean mask per gate."""
    h, overlap = F["height"], F["truck_overlap"]
    truck_gap, cabin_gap = side_signals(F)
    with np.errstate(invalid="ignore"):
        if phase == "digging":
            # not over the truck (within its left-right span, above its middle), and on
            # the pile side of it
            return [overlap <= 0, truck_gap < 0]
        if phase == "hauling":
            # above this cycle's dig height, and not over the truck
            return [h > dig_reference(t, F, anchor, cycle), overlap <= 0]
        if phase == "dumping":
            # on the truck side of the cabin, and above this cycle's dig height. No
            # "above the cabin" gate: on a low, far camera the bucket is level with the
            # cab while it dumps, so that gate held for one frame in 54 s.
            return [cabin_gap > 0, h > dig_reference(t, F, anchor, cycle)]
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


# Tier weights -- strong 3, medium 2, weak 1 -- agreed with Aanya on 2026-09-27
# ("some of these are a lot more valuable cues than others"). Set by judgment about
# which motions really mark a phase, informed by (not fitted to) how often each cue
# contained the marks: fitting them to these marks would only learn these clips.
# Cues that only showed up on THESE clips' graphs, with no physical reason they must
# happen on another video, do not vote and are not here (the universality audit;
# Aanya: "let's get rid of the only seen on graphs (radius) cues").
WEIGHTS = {
    ("digging", "speed_min"): 3,
    ("digging", "height_drop_end"): 3,
    ("digging", "x_drop_end"): 3,  # the bucket back at the pile side
    ("digging", "dx_crossing"): 2,
    ("digging", "dh_back_to_0"): 1,
    ("hauling", "height_takeoff"): 3,
    ("hauling", "truck_x_takeoff"): 3,
    ("dumping", "height_second_hump"): 2,
    ("dumping", "dh_near_0"): 1,
    ("dumping", "dx_positive"): 1,
    ("dumping", "overlapping"): 1,
    ("swinging", "x_takeoff"): 3,
    ("swinging", "dh_positive_falling"): 1,
    ("swinging", "height_peak"): 1,
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
        # the majority is out of ALL the supporting cues' weight, not only the cues
        # that found something -- one cue alone is not a majority
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
    # out of ALL the stage's cue weight, not only the cues that found something (a
    # lone weak cue used to count as a unanimous vote)
    # Haul keeps the count of the cues that found something: it has only two cues
    # and already REQUIRES its height climb, so one silent cue must not halve it.
    total = sum(x for _, x in present) if phase == "hauling" else sum(wt)
    count = np.zeros(len(t))
    for w, x in present:
        count += x * ((t >= w["window"][0] - 1e-9) & (t <= w["window"][1] + 1e-9))
    ok = (2 * count > total) & (t >= cue_from - 1e-9)
    if phase == "hauling":
        ok &= (t >= ra - 1e-9) & (t <= rb + 1e-9)
    runs = cues.runs(ok, t)
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
        runs = cues.runs(allowed & (count >= best - 1e-9), t)
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


def step(t, F, found, phase, anchor, cycle):
    """Search for one phase after ``anchor``.

    Returns {"phase", "window"} plus, when the vote ran, "core" (where the agreeing
    cues overlap), "weak" (no majority: the window is where the MOST weight agrees),
    and "votes"/"need" (the weight that agreed, and half the total). ``window`` is None
    when the gates never open or nothing agrees."""
    graphs = found[phase]
    cycle["search_from"] = anchor
    gate = np.ones(len(t), bool)
    for mask in gates(t, F, phase, anchor, cycle):
        gate &= mask
    open_idx = next((j for j in range(_i(t, anchor), len(t)) if gate[j]), None)
    if open_idx is None:
        return {"phase": phase, "window": None}
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
        visit = next(
            (v for v in cues.runs(F["truck_overlap"] > 0, t) if v[1] > cue_from), None
        )
        if visit is not None:
            cands = [e for e in cands if visit[0] - 1e-9 <= e["centre"] <= visit[1] + 1e-9]
        tried = []
        for e in cands:  # each candidate tip, with the checks read inside ITS window
            trial = list(windows)
            trial[k0] = widen(t, pg, e, cue_from)
            fill_checks(t, graphs, trial, trial[k0])
            got = combine(t, phase, graphs, trial, cue_from)
            tried.append((got[1], -e["centre"], got))
        if tried:  # the most support wins; ties go to the earlier candidate
            window, n, need, core = max(tried, key=lambda x: (x[0], x[1]))[2]
        else:
            window, n, need, core = None, 0, 0, None
    else:
        window, n, need, core = combine(t, phase, graphs, windows, cue_from)
    weak = window is not None and not (2 * n > 2 * need if need else True)
    return {
        "phase": phase,
        "window": window,
        "core": core,
        "weak": bool(weak),
        "votes": n,
        "need": need,
    }


PHASE_ORDER = ("digging", "hauling", "dumping", "swinging")


def dump_stretches(t, F):
    """Where the bucket is over the truck long enough to dump: overlap stretches at
    least half as long as the longest one in the video. The short passes over the
    truck at the start of each return swing (about 1 s here) do not count."""
    runs = cues.runs(F["truck_overlap"] > 0, t)
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


def settled_in_pile(t, F, a, b, levels):
    """WEAK dig signal, used only when the dig cues find nothing in the stretch: the
    first moment in (a, b) the bucket is low (within a quarter of the way from the
    stretch's pile level to its arrival height), on the pile side, off the truck and
    at rest (2D speed within 3x its noise, at least 2% of its range)."""
    m = (t >= a) & (t <= b)
    if not m.any():
        return None
    h, speed = F["height"], F["speed_2d"]
    p5, p95 = levels
    rest = max(3 * noise_scale(speed), 0.02 * float(np.nanmax(speed) - np.nanmin(speed)))
    low = p5 + 0.25 * (p95 - p5)
    with np.errstate(invalid="ignore"):
        ok = (
            m
            & (h <= low)
            & (side_signals(F)[0] < 0)
            & (F["truck_overlap"] <= 0)
            & (speed <= rest)
        )
    idx = np.flatnonzero(ok)
    return None if idx.size == 0 else float(t[idx[0]])


def find_dig(t, F, found, anchor, cycle, reach=2, weak=True):
    """A dig, in the open stretch the search is in -- or, if nothing there, the next
    one (Aanya: "if you can't find a cue for digging in that then look at the next one
    and it has to be within one of those two"). In each stretch: the dig cues first,
    then the weak "settled in the pile" signal, before moving on -- so a clip that
    opens mid-dig, with no return swing before it for the cues to read, keeps its
    first dig. Every dig ends before the stretch's climb to the truck."""
    stretches = open_stretches(t, F)
    near = [k for k, (a, b) in enumerate(stretches) if b > anchor + 1e-6][:reach]
    for k in near:
        a, b = stretches[k]
        lo = max(a, anchor)
        hi = climb_start(t, F["height"], a, b, last=k == len(stretches) - 1)
        if hi <= lo:
            continue
        s = step(t, F, found, "digging", lo, cycle)
        if s["window"] is not None:
            ca, cb = s.get("core") or s["window"]
            if lo - 1e-9 <= (ca + cb) / 2 <= hi:
                return {**s, "window": (s["window"][0], min(s["window"][1], hi))}
        if not weak:
            continue
        levels = pile_and_arrival(t, F["height"], a, b, last=k == len(stretches) - 1)
        c = settled_in_pile(t, F, lo, hi, levels)
        if c is not None:
            w = (max(c - MIN_HALF, lo), min(c + MIN_HALF, hi))
            return {"phase": "digging", "window": w, "core": w, "weak": True}
    return None


def from_found(t, F, side):
    """NO LABELS: each stage is searched for after the END of the window this code
    found for the stage before it -- dig, haul, dump, swing, round again -- from the
    clip's first frame (Aanya: "we need to make sure the condition of the end of the
    prev interval holds before looking for the next stage"). Digs are found by
    find_dig (inside an open stretch, before its climb to the truck).

    Recovery ("a strong dig abandons the cycle and restarts"): a STRONG dig is one the
    dig cues find, never the weak "settled in the pile" signal. If a stage finds
    nothing, or finds its window only after a strong dig, the cycle is abandoned, the
    search restarts at that dig, and the gap is recorded -- a skipped stage is
    reported, never guessed. It stops only when no strong dig is left.

    Returns (steps, stop, gaps): one step per phase found, why the search stopped (or
    None), and every abandoned cycle."""
    found = finders(t, F, side)
    out, cycle, anchor, k, stop, gaps = [], {}, float(t[0]), 0, None, []
    dig_k = PHASE_ORDER.index("digging")
    while anchor < t[-1]:
        phase = PHASE_ORDER[k % len(PHASE_ORDER)]
        if phase == "digging":
            cycle = {}
            s = find_dig(t, F, found, anchor, cycle)
            if s is None:
                s = find_dig(t, F, found, anchor, cycle, reach=None, weak=False)
                if s is not None:
                    gaps.append(
                        {
                            "phase": phase,
                            "from": anchor,
                            "to": s["window"][0],
                            "why": "no dig in the next two open stretches; restarted at "
                            "the next strong dig",
                        }
                    )
        else:
            s = step(t, F, found, phase, anchor, cycle)
            strong = find_dig(t, F, found, anchor, {}, reach=None, weak=False)
            missed = s["window"] is None
            late = (
                not missed
                and strong is not None
                and strong["window"][1] <= s["window"][0] + 1e-9
            )
            if not missed and not late and s.get("weak") and strong is None:
                # A low-agreement window is only accepted when a later dig shows the
                # cycle went on (Aanya: "if there was a weak signal it shouldn't have
                # been accepted unless we knew that there was a swing in that area
                # which would only happen if we knew there was a dig"). With no dig
                # after it, the clip may end before this stage happens.
                stop = {
                    "phase": phase,
                    "search_from": anchor,
                    "why": f"only a low-agreement window ({s['window'][0]:.1f} s) and no "
                    "dig after it: the clip may end before this stage",
                }
                break
            if missed or late:
                if strong is None:
                    stop = {"phase": phase, "search_from": anchor, "why": "no window found"}
                    break
                why = (
                    "no window found"
                    if missed
                    else f"its window ({s['window'][0]:.1f} s) came after a strong dig"
                )
                gaps.append(
                    {
                        "phase": phase,
                        "from": anchor,
                        "to": strong["window"][0],
                        "why": f"{why}; cycle abandoned, restarted at the dig",
                    }
                )
                cycle = {}
                s, phase, k = strong, "digging", dig_k
        if s["window"] is None:
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
    return out, stop, gaps
