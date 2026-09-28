# ruff: noqa: E501, RUF001
"""Draw the phase-window page from eval/interval_votes.py's output file(s).

    uv run python eval/window_page.py result.json [more.json ...] --out-dir pages/

Writes one HTML page per result file and prints a summary table. Nothing is computed
here except pixel positions and the comparison with the labelled marks (if any):
every window, cue and gate comes from the result file.
"""

import argparse
import html
import json
from pathlib import Path

D: dict = {}
t: list = []
T0 = T1 = 0.0
COL = {
    "digging": "var(--dig)",
    "hauling": "var(--haul)",
    "dumping": "var(--dump)",
    "swinging": "var(--swing)",
}
NAME = {"digging": "Dig", "hauling": "Haul", "dumping": "Dump", "swinging": "Return swing"}
W, L, R = 940, 52, 14

# Aanya's own words for each cue -- what each rule is built from.
QUOTES = {
    (
        "digging",
        "speed_min",
    ): "“Speed 2D: when you're looking at a big peak, it should be a big peak, not two tiny peaks … it's at the minimum of the peak, and then it starts going up. Make sure the window is around that minimum.” Fixed to the FIRST minimum (“it missed the first hump”).",
    (
        "digging",
        "height_drop_end",
    ): "“It should be directly after that big drop … It doesn't need to be a flat knee.” and “Height looks good.”",
    (
        "digging",
        "dx_crossing",
    ): "“the window should include the little peak and then the drop again that happens right after that. It should include the part that crosses the x-axis.” (0 is reached within the signal's noise.)",
    ("digging", "dh_back_to_0"): "Your key, dh/dt at dig: “negative approaching 0”.",
    (
        "digging",
        "radius_turn",
    ): "Your key, radius at dig: “after big dip and upswing downward trend”.",
    (
        "digging",
        "x_drop_end",
    ): "Your key, bucket x at dig: “just dropped a lot and now its increasing”, and “the truck and relative distance are valuable for detection, but maybe the x-coordinate provides that same value for digging”. The window starts where the bucket is back at the pile side.",
    (
        "hauling",
        "height_takeoff",
    ): "“draw a line from the tallest height in dig window to the graph for height and then the window starts there … a second long or something”, “make sure it keeps goign generally up after that”, “height is a rquirement”.",
    (
        "hauling",
        "dx_min",
    ): "“bottom of the dip with a spike after” and “it should include the minimum after the fluctuation. It's more like a plateau, and then there's a minimum.”",
    (
        "hauling",
        "truck_x_takeoff",
    ): "Your key: “negative, inflection point”; “widen to not just start at the knee but consider more of the knee.”",
    (
        "hauling",
        "radius_dip_start",
    ): "Your key, radius at haul: “start of dip”. Fixed: “There are pretty regular V-shaped triangle dips, so what happened?” — it now finds the start of each big V dip (a knee from flat found almost nothing).",
    (
        "dumping",
        "aspect_steepest",
    ): "“The window for dump should include the steepest drops in aspect ratio … it needs to be very validated by some of the other things.” and “You can detect them multiple times, and then you can figure out which is the real signal based on validating across multiple.”",
    (
        "dumping",
        "cabin_x_going_up",
    ): "“I think relative cabin exposition [x position] should be going up.” Your key: “down bump and now its going up again”.",
    (
        "dumping",
        "height_second_hump",
    ): "“it should be in the middle. It shouldn't contain the maximum. I was just saying that you need to know that there is a maximum coming.” and “make this in the middle, like it's in the valley”: centred on the bottom of the valley, where the climb into the second hump begins.",
    ("dumping", "dh_near_0"): "Your key, dh/dt at dump: “close to or at 0”.",
    ("dumping", "dx_positive"): "Your key, dx/dt at dump: “positive”.",
    ("dumping", "overlapping"): "Your key, bucket − truck x at dump: “overlap”.",
    ("swinging", "x_takeoff"): "Your key, bucket x: “start right before peak”.",
    ("swinging", "overlap_end"): "Your key: “end of overlap”.",
    (
        "swinging",
        "height_peak",
    ): "Your key, height: “in peak”. And: “shouldn't more heights be identified, and then from that, we choose what the ideal height is? … make this more durable for understanding variations and not just looking for the most specific patterns, but rather looking at things consistently.” Shaded: every candidate top (the part within 20% of it). Chosen: the last top before the big height drop, placed where the climb gets within 20% of it.",
    ("swinging", "dh_positive_falling"): "Your key, dh/dt: “positive decreasing”.",
    (
        "swinging",
        "dx_bump",
    ): "Your key, dx/dt: “upward bump right after”. Centred where the bump starts, not at its top.",
    (
        "swinging",
        "radius_bump_start",
    ): "Your key, radius: “start of bump after”. Fixed (“how come nothing was detected?”): where the steep part of the big bump begins.",
}
ORDER_TEXT = {
    "digging": (
        "no truck overlap · on the pile side of the truck",
        "none",
        "more than half of the cues' total weight must agree (strong 3, medium 2, weak 1)",
    ),
    "hauling": (
        "height above this cycle's dig height · no truck overlap",
        "height (it must be part of the window)",
        "more than half of the total weight of the cues within 5 s of height's window",
    ),
    "dumping": (
        "on the truck side of the cabin · height above the dig height · above the cabin",
        "every aspect-ratio drop of ≥ 30% is a candidate; its steepest point ± uncertainty is the window",
        "the candidate with the most supporting weight is kept (flagged low agreement if that is not more than half of the others' weight)",
    ),
    "swinging": ("none", "none", "more than half of the cues' total weight must agree"),
}


def X(s):
    return L + (s - T0) / (T1 - T0) * (W - L - R)


def ticks(y0, y1, H):
    return "".join(
        f'<line x1="{X(s):.1f}" x2="{X(s):.1f}" y1="{y0}" y2="{y1}" stroke="var(--grid)"/>'
        f'<text x="{X(s):.1f}" y="{H - 6}" class="tk" text-anchor="middle">{s}s</text>'
        for s in range(0, int(T1) + 1, 10)
    )


def trace(key, H, TOP, BOT):
    v = [x for x in D["signals"][key] if x is not None]
    vs = sorted(v)
    lo, hi = vs[int(0.01 * len(vs))], vs[int(0.99 * len(vs)) - 1]
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad
    if hi <= lo:  # a flat signal (e.g. overlap 0 throughout): centre it, don't divide by 0
        lo, hi = lo - 1, hi + 1
    Y = lambda y: TOP + (1 - (min(max(y, lo), hi) - lo) / (hi - lo)) * (H - TOP - BOT)  # noqa: E731
    pts = " ".join(
        f"{X(a):.1f},{Y(b):.1f}"
        for a, b in zip(t, D["signals"][key], strict=True)
        if b is not None
    )
    zero = (
        f'<line x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--rule)"/>'
        if lo < 0 < hi
        else ""
    )
    return (
        zero
        + f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="1.2" stroke-linejoin="round"/>'
    )


def marks_svg(marks, TOP, H, BOT):
    return "".join(
        f'<line x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{TOP}" y2="{H - BOT}" stroke="var(--truth)" stroke-width="1.8" stroke-dasharray="4 3"/>'
        for m in marks
    )


def weak_graph(ci, steps, marks):
    """One condition of the weak dig signal: shaded where it holds inside the search
    region, and ▾ where all four first hold (the chosen moment)."""
    weak = [s for s in steps if s.get("weak_signal")]
    c0 = weak[0]["weak_signal"]["conditions"][ci]
    H, TOP, BOT = 110, 12, 22
    o = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="weak signal {html.escape(c0["id"])}">',
        ticks(TOP, H - BOT, H),
    ]
    for st in weak:
        for a, b in st["weak_signal"]["conditions"][ci]["runs"]:
            o.append(
                f'<rect x="{X(a):.1f}" y="{TOP}" width="{max(X(b) - X(a), 1):.1f}" height="{H - TOP - BOT}" fill="var(--c)" fill-opacity=".22"/>'
            )
        x = X(st["weak_signal"]["at"])
        o.append(f'<path d="M{x:.1f},{TOP + 1} l-5,-7 h10 z" fill="var(--c)"/>')
    o.append(trace(c0["key"], H, TOP, BOT))
    o.append(marks_svg(marks, TOP, H, BOT))
    o.append("</svg>")
    return (
        f"<figure><figcaption><b>{html.escape(c0['key'])}</b> · {html.escape(c0['id'])}: {html.escape(c0['label'])} "
        f'<span class="dim">(shaded = holds inside the search region, ▾ = all four first hold)</span></figcaption><div class="plot">{"".join(o)}</div></figure>'
    )


def gate_graph(_phase, gi, steps, marks):
    g0 = steps[0]["gates"][gi]
    H, TOP, BOT = 110, 12, 22
    o = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="gate {g0["id"]}">',
        ticks(TOP, H - BOT, H),
    ]
    for st in steps:
        if not st["gates"]:
            continue
        for a, b in st["gates"][gi]["runs"]:
            a, b = max(a, st["search_from"]), b
            if b > a:
                o.append(
                    f'<rect x="{X(a):.1f}" y="{TOP}" width="{max(X(b) - X(a), 1):.1f}" height="{H - TOP - BOT}" fill="var(--gate)" fill-opacity=".22"/>'
                )
        if st.get("gates_open") is not None:
            x = X(st["gates_open"])
            o.append(f'<path d="M{x:.1f},{TOP + 1} l-5,-7 h10 z" fill="var(--gate)"/>')
        o.append(
            f'<path d="M{X(st["search_from"]):.1f},{TOP - 2} l6,-5 v10 z" fill="var(--muted)"/>'
        )
    o.append(trace(g0["key"], H, TOP, BOT))
    o.append(marks_svg(marks, TOP, H, BOT))
    o.append("</svg>")
    return (
        f"<figure><figcaption><b>{html.escape(g0['key'])}</b> · gate {g0['id']}: {html.escape(g0['label'])} "
        f'<span class="dim">(shaded = holds, ▾ = all gates open)</span></figcaption><div class="plot">{"".join(o)}</div></figure>'
    )


def cue_graph(phase, fi, finder, steps, marks):
    key = finder["signal"]
    H, TOP, BOT = 140, 12, 22
    c = COL[phase]
    o = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(finder["label"])}">',
        ticks(TOP, H - BOT, H),
    ]
    for a, b in finder["spans"]:
        o.append(
            f'<rect x="{X(a):.1f}" y="{TOP}" width="{max(X(b) - X(a), 1.5):.1f}" height="{H - TOP - BOT}" fill="{c}" fill-opacity=".12"/>'
        )
    for st in steps:
        gw = st["graphs"][fi]
        if not gw:
            continue
        if gw.get("context"):
            a, b = gw["context"]
            o.append(
                f'<rect x="{X(a):.1f}" y="{TOP + 1}" width="{max(X(b) - X(a), 1.5):.1f}" height="{H - TOP - BOT - 2}" fill="none" stroke="var(--muted)" stroke-dasharray="3 3"/>'
            )
        a, b = gw["window"]
        o.append(
            f'<rect x="{X(a):.1f}" y="{TOP}" width="{max(X(b) - X(a), 2):.1f}" height="{H - TOP - BOT}" fill="{c}" fill-opacity=".5" stroke="{c}" stroke-width="{2.2 if gw.get("capped") else 1}"/>'
        )
        if gw.get("half") is not None:
            o.append(
                f'<line x1="{X(gw["centre"]):.1f}" x2="{X(gw["centre"]):.1f}" y1="{TOP}" y2="{TOP + 10}" stroke="var(--ink)" stroke-width="2"/>'
            )
    o.append(trace(key, H, TOP, BOT))
    for st in steps:
        if st.get("cue_from") is not None:
            o.append(
                f'<path d="M{X(st["cue_from"]):.1f},{TOP - 2} l6,-5 v10 z" fill="var(--muted)"/>'
            )
    o.append(marks_svg(marks, TOP, H, BOT))
    o.append("</svg>")
    quote = QUOTES.get((phase, finder["id"]), "")
    role = {"required": "REQUIRED · ", "primary": "PRIMARY · "}.get(finder.get("role"), "")
    if finder.get("role") != "primary":
        role += f"weight {finder.get('weight', 1)} · "

    kind = " (checked inside the aspect window)" if finder.get("mode") == "check" else ""
    return (
        f"<figure><figcaption><b>{html.escape(key)}</b> · {role}{html.escape(finder['label'])}{kind}</figcaption>"
        f'{f"<p class=quote>{html.escape(quote)}</p>" if quote else ""}<div class="plot">{"".join(o)}</div></figure>'
    )


def window_strip(phase, steps, marks):
    H = 46
    c = COL[phase]
    o = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="phase windows">',
        ticks(4, H - 18, H),
        f'<text x="{L - 6}" y="20" class="tk" text-anchor="end">window</text>',
    ]
    for st in steps:
        if st["window"]:
            a, b = st["window"]
            if st.get("weak"):
                o.append(
                    f'<rect x="{X(a):.1f}" y="6" width="{max(X(b) - X(a), 2.5):.1f}" height="20" fill="{c}" fill-opacity=".35" stroke="{c}" stroke-width="1.5" stroke-dasharray="4 2"/>'
                )
            else:
                o.append(
                    f'<rect x="{X(a):.1f}" y="6" width="{max(X(b) - X(a), 2.5):.1f}" height="20" fill="{c}" stroke="{c}"/>'
                )
        else:
            o.append(
                f'<text x="{X(st["search_from"]) + 8:.1f}" y="20" class="tk none">no window</text>'
            )
    o.append(marks_svg(marks, 2, H, 18))
    o.append("</svg>")
    return f'<figure><figcaption><b>Phase window</b> · the result (dashed = low agreement: no location had more than half the weight, so the window is where the most weight agrees — “there has to be something for that given stage transition”)</figcaption><div class="plot">{"".join(o)}</div></figure>'


def table(_phase, steps, marks, finders):
    rows = []

    # marks are only compared: each found window is paired with the nearest mark of
    # the same stage (the search never reads them)
    def near(st):
        w = st["window"] or (st["search_from"], st["search_from"])
        return min(marks, key=lambda m: max(w[0] - m, m - w[1], 0.0)) if marks else None

    for st in steps:
        m = near(st)
        w = st["window"]
        found = sum(g is not None for g in st["graphs"])
        if st.get("how", "").startswith("settled"):
            r = st["region"]
            how = f"weak signal: settled in the pile (no dig cue fired in {r[0]:.1f}–{r[1]:.1f} s)"
        elif st.get("clip_start") is not None:
            how = "clip-start rule"
        else:
            how = (
                (
                    f"searched {st['region'][0]:.1f}–{st['region'][1]:.1f} s (ends where the climb to the truck starts) · "
                    if st.get("region")
                    else ""
                )
                + f"weight {st['votes']:.0f} agrees (more than half is {st['need']:.1f}); {found}/{len(finders)} cues found"
            )
            if st.get("weak"):
                how += ' · <span class="res bad">low agreement: where the most weight agrees</span>'
        if w and m is None:
            res, win, wid = (
                '<span class="dim">no label</span>',
                f"{w[0]:.2f}–{w[1]:.2f} s",
                f"{w[1] - w[0]:.2f} s",
            )
        elif w:
            ok = w[0] <= m <= w[1]
            gap = 0 if ok else (w[0] - m if m < w[0] else m - w[1])
            res = (
                '<span class="res good">contains</span>'
                if ok
                else f'<span class="res bad">{"starts" if m < w[0] else "ends"} {abs(gap):.2f} s {"after" if m < w[0] else "before"}</span>'
            )
            win, wid = f"{w[0]:.2f}–{w[1]:.2f} s", f"{w[1] - w[0]:.2f} s"
        else:
            res, win, wid = '<span class="res bad">no window</span>', "—", "—"
        go = "—" if st.get("gates_open") is None else f"{st['gates_open']:.2f} s"
        rows.append(
            f"<tr><td>{'—' if m is None else f'{m:.2f} s'}</td><td>{st['search_from']:.2f} s</td><td>{go}</td><td>{how}</td><td>{win}</td><td>{wid}</td><td>{res}</td></tr>"
        )
    return (
        "<div class='tbl'><table><thead><tr><th>your mark</th><th>search from</th><th>gates open</th><th>cues</th>"
        "<th>window</th><th>width</th><th>vs your mark</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def horizontal_graph(key, title, note, dips=None, extra=None):
    """One plain feature trace with every mark (coloured by phase)."""
    H, TOP, BOT = 120, 14, 22
    o = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
        ticks(TOP, H - BOT, H),
    ]
    for m in D["marks"]:
        o.append(
            f'<line x1="{X(m["t"]):.1f}" x2="{X(m["t"]):.1f}" y1="{TOP}" y2="{H - BOT}" stroke="{COL[m["phase"]]}" stroke-width="1.6" stroke-dasharray="4 3"/>'
        )
    o.append(trace(key, H, TOP, BOT))
    for d in dips or []:
        bad = extra is not None and d in extra
        c = "var(--bad)" if bad else "var(--ink)"
        o.append(f'<path d="M{X(d):.1f},{H - BOT + 1} l-5,-8 h10 z" fill="{c}"/>')
    o.append("</svg>")
    return (
        f'<figure><figcaption><b>{html.escape(title)}</b> <span class="dim">{note}</span></figcaption>'
        f'<div class="plot">{"".join(o)}</div></figure>'
    )


def render():
    """The whole page for the loaded result file D."""
    hd = D["horizontal_dips"]
    extra = [
        d
        for d in hd["toward_dist_dt"]
        if min((abs(d - e) for e in hd["toward_dx_dt"]), default=9) > 1.0
    ]
    legend = " ".join(
        f'<span><i style="border-top:2px dashed {COL[p]};width:18px;display:inline-block;vertical-align:middle"></i> {NAME[p]} mark</span>'
        for p in COL
    )
    horizontal = f"""<section class="phase" style="--c:var(--muted)"><h2>Bucket x vs distance vs the pile-to-truck line</h2>
    <p class="note">“can you add distance as a feature and put it on the artifact so i can see this visually”. Distance = √(x gap² + y gap²) from the bucket to the truck box's centre, in arm reaches L. The x gap keeps which SIDE of the truck the bucket is on; distance does not. “the goal is to get rid of the position specific ones”: the pile-to-truck position measures where the bucket is along the line from the pile (found as where the bucket sits at its lowest) to the truck, so it keeps the side without assuming the camera sees them left–right. Right after each swing mark the bucket moves past the truck's centre: the x gap keeps rising, while distance grows as if the bucket were already heading back to the pile.</p>
    <div class="key">{legend}<span>▾ big dip the dig “crosses back to 0” cue can pick</span><span style="color:var(--bad)">▾ extra dip only distance has</span></div>
    {horizontal_graph("pile_truck_pos", "pile-to-truck position (NOW USED by every horizontal cue and gate)", f"· −1 = at the pile, 0 = at the truck's centre, + = past it. The line's direction is found from the video: tilted {D['pile_truck_angle']:.0f}° from level here.")}
    {horizontal_graph("rel_truck_x", "x gap to the truck (bucket − truck x, + toward the truck)", "· keeps the side")}
    {horizontal_graph("truck_distance", "distance to the truck", "· L, always ≥ 0: the side is lost")}
    {horizontal_graph("toward_dx_dt", "x speed toward the truck (dx/dt, + toward)", "· one big dip per cycle: the rush back to the pile", hd["toward_dx_dt"])}
    {horizontal_graph("toward_dist_dt", "distance speed toward the truck (− d distance/dt, + toward)", "· red = dips that x speed does not have (on the labelled clips: the swing moving past the truck); the dig cue takes the FIRST dip after the swing", hd["toward_dist_dt"], extra)}
    </section>"""

    stop = D.get("stop")
    stop_note = (
        f"<p class='note'>The search stopped at {NAME[stop['phase']].lower()}, searched from {stop['search_from']:.2f} s: {stop['why']} (the clip ends {T1 - stop['search_from']:.1f} s later, before another {NAME[stop['phase']].lower()} starts).</p>"
        if stop
        else ""
    )
    sections = [horizontal]
    for phase in ("digging", "hauling", "dumping", "swinging"):
        steps = [s for s in D["steps"] if s["phase"] == phase]
        marks = [m["t"] for m in D["marks"] if m["phase"] == phase]
        finders = D["finders"][phase]
        gates_txt, req_txt, sup_txt = ORDER_TEXT[phase]
        gated = next((s for s in steps if s["gates"]), None)
        weak_steps = [s for s in steps if s.get("weak_signal")]
        weak_figs = (
            "<h3>Weak signal · settled in the pile</h3><p class='note'>Used only when no dig cue fires in the search region — here because the clip opens mid-dig, with no return swing before it for the dig cues to read. The window is the first moment all four hold, ± 0.75 s. “If you're at the start of a video, maybe have weaker signals.”</p>"
            + "".join(
                weak_graph(ci, weak_steps, marks)
                for ci in range(len(weak_steps[0]["weak_signal"]["conditions"]))
            )
            if weak_steps
            else ""
        )
        gate_figs = (
            "".join(
                gate_graph(phase, gi, [s for s in steps if s["gates"]], marks)
                for gi in range(len(gated["gates"]))
            )
            if gated
            else "<p class='dim'>No gates for this phase.</p>"
        )
        lead = [i for i, f in enumerate(finders) if f.get("role") in ("required", "primary")]
        rest = [i for i, f in enumerate(finders) if i not in lead]
        lead_figs = (
            "".join(cue_graph(phase, i, finders[i], steps, marks) for i in lead)
            or "<p class='dim'>No required cue for this phase.</p>"
        )
        rest_figs = "".join(cue_graph(phase, i, finders[i], steps, marks) for i in rest)
        note = ""
        if phase == "digging":
            note = "<p class='note'>Each dig is searched for inside an <b>open stretch</b>: between two stretches where the bucket is over the truck long enough to dump (at least half as long as the longest one). The search ends where that stretch's <b>climb to the truck starts</b> (height walked back from halfway between pile level and arrival height), so a dig window can never reach into the haul. Order: the dig cues first; if none fire, the weak signal “settled in the pile” (low, pile side, off the truck, at rest); only then the next open stretch (“if you can't find a cue for digging in that then look at the next one”). The haul is searched from the end of the dig window. No fixed 5 s limit any more.</p>"
        if phase == "dumping":
            note = "<p class='note'>Dump windows mark the START OF TIPPING (the definition agreed on 2026-09-27), so label dumps at the start of tipping, not at arrival over the truck.</p>"
        sections.append(f"""<section class="phase" style="--c:{COL[phase]}"><h2><span class="dot"></span>{NAME[phase]}</h2>
    <ol class="order"><li><b>Gates</b> — nothing below is considered until these hold: {html.escape(gates_txt)}</li>
    <li><b>Required cue</b> — {html.escape(req_txt)}</li><li><b>Supporting cues</b> — {html.escape(sup_txt)}; the window is then the full extent of the cues that agree, at most 2.4 s</li></ol>
    <h3>1 · Gates</h3>{gate_figs}{weak_figs}<h3>2 · Required cue</h3>{lead_figs}<h3>3 · Supporting cues</h3>{rest_figs}
    {window_strip(phase, steps, marks)}{table(phase, steps, marks, finders)}{note}</section>""")

    page = f"""<title>{html.escape(D["clip"])} phase windows</title>
    <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
    <style>
    :root{{--bg:#f4f5f2;--panel:#fff;--ink:#1d2320;--muted:#66706a;--line:#dde1dc;--grid:#eceee9;--rule:#b8bfb9;--truth:#c2410c;--gate:#6d5bd0;
    --good:#1f7a4a;--good-bg:#dff1e6;--bad:#9a3412;--bad-bg:#fde8dc;--dig:#c98400;--haul:#2f86c2;--dump:#00866a;--swing:#b0588f}}
    @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#111513;--panel:#181d1a;--ink:#e6ebe7;--muted:#95a09a;--line:#2b332e;--grid:#1f2622;--rule:#46504a;--truth:#fb923c;--gate:#a99bff;
    --good:#7fd8a4;--good-bg:#173726;--bad:#fdba74;--bad-bg:#3b2414;--dig:#E69F00;--haul:#56B4E9;--dump:#2fbf95;--swing:#d98bbb}}}}
    :root[data-theme="dark"]{{color-scheme:dark;--bg:#111513;--panel:#181d1a;--ink:#e6ebe7;--muted:#95a09a;--line:#2b332e;--grid:#1f2622;--rule:#46504a;--truth:#fb923c;--gate:#a99bff;
    --good:#7fd8a4;--good-bg:#173726;--bad:#fdba74;--bad-bg:#3b2414;--dig:#E69F00;--haul:#56B4E9;--dump:#2fbf95;--swing:#d98bbb}}
    body{{background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif}}
    .wrap{{max-width:1040px;margin:0 auto;padding-inline:16px;padding-block:28px 48px;display:grid;gap:22px}}
    h1{{font-size:22px;font-weight:600;margin:0}}.sub{{color:var(--muted);margin:6px 0 0;max-width:84ch}}
    h3{{margin:10px 0 0;font-size:12.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
    .key{{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:12.5px;color:var(--muted);align-items:center}}
    .key i{{display:inline-block;width:22px;height:11px;vertical-align:-1px;margin-right:6px;border-radius:2px}}
    .k1{{background:var(--muted);opacity:.25}}.k2{{background:var(--muted);opacity:.7}}.k3{{height:0!important;border-top:2px dashed var(--truth)}}.k4{{border:1px dashed var(--muted);height:9px!important}}.k5{{background:var(--gate);opacity:.3}}
    .phase{{background:var(--panel);border:1px solid var(--line);border-top:3px solid var(--c);border-radius:6px;padding:16px 18px;display:grid;gap:10px}}
    .phase h2{{margin:0;font-size:18px;font-weight:600;display:flex;align-items:center;gap:9px}}.dot{{width:11px;height:11px;border-radius:50%;background:var(--c)}}
    .order{{margin:0;padding-left:20px;font-size:13px;display:grid;gap:2px}}
    figure{{margin:0}}figcaption{{font-size:12.5px;color:var(--muted);margin-bottom:2px}}figcaption b{{font:500 12.5px "IBM Plex Mono",monospace;color:var(--ink)}}
    .dim{{color:var(--muted);font-size:12px}}
    .quote{{margin:2px 0 4px;font-size:12.5px;font-style:italic;color:var(--ink);border-left:3px solid var(--c);padding-left:9px;max-width:92ch}}
    .plot{{overflow-x:auto}}.plot svg{{display:block;width:100%;min-width:620px;height:auto}}
    .tk{{font:400 10.5px "IBM Plex Mono",monospace;fill:var(--muted)}}.none{{fill:var(--bad)}}
    .tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;min-width:760px}}
    th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-weight:500}}
    .res{{font:500 11.5px "IBM Plex Mono",monospace;padding:2px 6px;border-radius:3px;white-space:nowrap}}.good{{background:var(--good-bg);color:var(--good)}}.bad{{background:var(--bad-bg);color:var(--bad)}}
    .note{{margin:0;font-size:12.5px;color:var(--muted);border-left:2px solid var(--line);padding-left:10px;max-width:86ch}}
    </style>
    <div class="wrap">
    <header><h1>Phase interval votes</h1>
    <p class="sub">{html.escape(D["clip"])}. <b>No labels are used.</b> Each stage is searched for after the window the code found for the stage before it (▸ = where each search starts: the middle of the previous stage’s window), starting at the clip’s first frame; a stage is never skipped. “every single convo has been about making it run without you messing with it”, in this order: <b>1 · gates</b> — necessary thresholds; no cue is considered until they hold · <b>2 · the required cue</b> · <b>3 · supporting cues</b> — weighted: more than half of the total weight of the cues that found something must agree (strong 3, medium 2, weak 1), and the window is then the full extent of the agreeing cues. A cue's own window is its event's centre (tick) ± its timing uncertainty, clamped to ±0.75–1.2 s. Cues seen only on these graphs, with no physical reason to expect them on another video (all radius cues, cabin x going up, haul dx minimum, swing dx bump), no longer vote: “let’s get rid of the only seen on graphs (radius) cues”. Everything is computed by <code>eval/interval_votes.py</code>; your marks (dashed) are only compared with the result, never read by the search.</p></header>
    <div class="key"><span><i class="k5"></i>where a gate holds</span><span>▾ all gates open</span><span><i class="k1"></i>everywhere a cue's pattern occurs</span><span><i class="k2"></i>the cue's window that counts, per cycle</span><span><i class="k4"></i>context (the hump, drop or dip read first)</span><span><i class="k3"></i>your mark</span></div>
    {stop_note}{"".join(sections)}
    </div>"""
    return page


def summary(name, data):
    """One line per stage: labelled marks inside a window, mean window width."""
    rows = []
    for phase in ("digging", "hauling", "dumping", "swinging"):
        steps = [s for s in data["steps"] if s["phase"] == phase and s["window"]]
        widths = [s["window"][1] - s["window"][0] for s in steps]
        sc = data.get("score") or {}
        got = [r for r in sc.get("rows", []) if r["phase"] == phase]
        inside = f"{sum(r['inside'] for r in got)}/{len(got)}" if got else "no labels"
        mean = f"{sum(widths) / len(widths):.2f} s" if widths else "-"
        low = sum(bool(s.get("weak")) for s in steps)
        rows.append(
            f"  {name:<24} {phase:<10} windows {len(steps):>2}  marks inside {inside:<9} mean width {mean:<7} low agreement {low}"
        )
    stop = data.get("stop")
    if stop:
        rows.append(
            f"  {name:<24} stopped at {stop['phase']} (from {stop['search_from']:.2f} s): {stop['why']}"
        )
    return rows


def main(argv=None) -> int:
    global D, t, T0, T1
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, nargs="+", help="interval_votes.py --out files")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in args.results:
        D = json.loads(path.read_text(encoding="utf-8"))
        t = D["times"]
        T0, T1 = t[0], t[-1]
        out = args.out_dir / (path.stem + ".html")
        out.write_text(render(), encoding="utf-8")
        print(f"  wrote {out}")
        lines += summary(D["clip"], D)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
