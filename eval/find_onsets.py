"""Turn each phase window from eval/interval_votes.py into one start time.

    uv run python eval/find_onsets.py result.json --clip long --out onsets.json
    uv run python eval/find_onsets.py result.json --features D/features.npz --out onsets.json

The window says roughly where a phase starts; this picks the moment inside it:

    dig    the lowest 2D speed inside the dig window (Aanya: "for now for dig just
           find the local min of speed in that interval")
    haul   when the height starts rising               (the bucket starts lifting)
    dump   when the aspect ratio starts falling        (the bucket starts tipping)
    swing  when it starts moving along the pile-truck line, either way (the swing begins)

Each of the last three starts at the fastest movement in that direction inside the
window, then walks back while the rate is still moving that way faster than its own
noise (3x the noise measured in the window, as ``onsets.motion_boundary``). The
start is always inside the window (Aanya: "the specific frame has to be within
the interval"); if the movement is still going at the window's start, the start
is the window's start. When nothing in the window moves faster than its noise,
the start is the window's centre, flagged (Aanya: "it needs to report a frame in
the window so either it finds one or just the center of the window").

No labels are read to find anything. Given labels, each start is compared with the
nearest mark of its phase, next to the window's own start for reference.
"""

import argparse
import json
from pathlib import Path

import check_cues as cc
import numpy as np

from excavator_cycles.onsets import noise_scale

SIGMA = 3.0  # "moving" = beyond 3x the rate's own noise (as onsets.motion_boundary)

# phase -> (the rate, the direction that phase moves it in (0: either), what it means)
RATE = {
    "hauling": ("dh_dt", +1, "the lift begins (height rising)"),
    "dumping": ("aspect_ratio_dt", -1, "the tip begins (aspect ratio falling)"),
    # either way: on the long clip the bucket first moves PAST the truck, away from
    # the pile (the open "swing overshoot" question), so no direction is assumed
    "swinging": (
        "pile_truck_pos_dt",
        0,
        "the swing begins (moving along the pile-truck line)",
    ),
}


def movement_start(rate, t, bracket, window, direction):
    """Walk back from the fastest movement in the window to where it began.

    "Began" = the rate, signed so this phase's movement is positive, was last no
    faster than its own noise band. Why not onsets.motion_boundary's "back to
    rest": the bucket changes shape and height all through the haul, so the tip
    begins out of motion, not out of stillness -- a walk that waits for rest slid
    back 4-10 s into the haul on two of three dumps.
    """
    signed = np.abs(rate) if direction == 0 else rate * direction
    inb = np.where((t >= bracket[0]) & (t <= bracket[1]) & np.isfinite(signed))[0]
    inw = inb[(t[inb] >= window[0]) & (t[inb] <= window[1])]
    if len(inb) < 5 or not len(inw):
        return None, None, None
    band = SIGMA * noise_scale(signed[inb])
    i = inw[np.argmax(signed[inw])]
    peak = float(t[i])
    if not np.isfinite(band) or signed[i] <= band:
        return None, band, peak  # nothing in the window moves this way faster than noise
    while i > inb[0] and np.isfinite(signed[i - 1]) and signed[i - 1] > band:
        i -= 1
    return float(t[i]), float(band), peak


def onsets(t, F, steps):
    """One start per window, always inside that window."""
    out = []
    band = peak = None
    for s in steps:
        w = s["window"]
        if w is None:
            out.append({"phase": s["phase"], "t": None, "how": "no window"})
            continue
        if s["phase"] == "digging":
            inside = np.where((t >= w[0]) & (t <= w[1]) & np.isfinite(F["speed_2d"]))[0]
            at = float(t[inside[np.argmin(F["speed_2d"][inside])]]) if len(inside) else None
            how = "lowest 2D speed in the window"
            bracket = list(w)
        else:
            key, direction, how = RATE[s["phase"]]
            bracket = list(w)
            at, band, peak = movement_start(F[key], t, bracket, w, direction)
        fallback = at is None
        if fallback:  # every window reports a frame: its centre when nothing is found
            at = (w[0] + w[1]) / 2
            how = "window centre (no movement faster than noise in the window)"
        out.append(
            {
                "phase": s["phase"],
                "t": at,
                "how": how,
                "fallback": fallback,
                "bracket": bracket,
                "band": band if s["phase"] != "digging" else None,
                "peak": peak if s["phase"] != "digging" else None,
            }
        )
    return out


def _clean(v):
    return None if not np.isfinite(v) else round(float(v), 5)


def compare(found, steps, marks):
    """Each start next to the nearest mark of its phase, and the window's own start."""
    rows = []
    for o, s in zip(found, steps, strict=True):
        ms = [m for p, m in marks if p == o["phase"]]
        if not ms or o["t"] is None:
            rows.append({**o, "mark": None})
            continue
        m = min(ms, key=lambda v: abs(v - o["t"]))
        rows.append(
            {**o, "mark": m, "error": o["t"] - m, "window_start_error": s["window"][0] - m}
        )
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("result", type=Path, help="eval/interval_votes.py's output")
    ap.add_argument(
        "--clip", choices=sorted(cc.CLIPS), help="a fixture clip's features and labels"
    )
    ap.add_argument("--features", type=Path, help="any pipeline features.npz")
    ap.add_argument("--labels", type=Path, help="labels, used only to score (optional)")
    ap.add_argument(
        "--out", type=Path, help="write the result file again, with the starts added"
    )
    args = ap.parse_args(argv)
    if args.clip:
        feat_path, label_path = cc.CLIPS[args.clip]
    elif args.features:
        feat_path, label_path = args.features, args.labels
    else:
        raise SystemExit("give --clip or --features")

    t, F = cc.load_features(feat_path)
    F["aspect_ratio_dt"] = cc.derivative(F["aspect_ratio"], t)
    data = json.loads(args.result.read_text())
    marks = cc.load_onsets(label_path) if label_path else []
    rows = compare(onsets(t, F, data["steps"]), data["steps"], marks)

    errs = []
    for r in rows:
        at = "  --  " if r["t"] is None else f"{r['t']:6.2f}"
        line = f"{r['phase']:<10} start {at}  ({r['how']})"
        if r["mark"] is not None:
            line += f"   mark {r['mark']:6.2f}  error {r['error']:+.2f}"
            line += f"   window start {r['window_start_error']:+.2f}"
            errs.append((r["phase"], r["error"], r["window_start_error"]))
        print(line)
    if errs:
        print("\nmean |error| per phase: start found here vs. the window's start")
        for phase in cc.PHASES:
            e = [(a, b) for p, a, b in errs if p == phase]
            if e:
                a, b = np.mean([abs(x) for x, _ in e]), np.mean([abs(y) for _, y in e])
                print(f"  {phase:<10} {a:.2f} s vs {b:.2f} s   ({len(e)} marks)")
    if args.out:
        data["onsets"] = rows
        # the curve each start was read from, signed so the phase's movement is +
        data["signals"]["onset_digging"] = [_clean(v) for v in F["speed_2d"]]
        for phase, (key, direction, _) in RATE.items():
            curve = np.abs(F[key]) if direction == 0 else F[key] * direction
            data["signals"][f"onset_{phase}"] = [_clean(v) for v in curve]
        args.out.write_text(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
