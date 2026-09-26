#!/usr/bin/env python3
"""DIAGNOSTIC, not pipeline. Where did pass 1 think each transition happened?

    ./scripts/windows          # the whole loop: run, print, render, open

Re-run it after every change to a cue. It is meant to be the inner loop of
working on the triggers, so it takes about five seconds and needs no arguments.

Not imported by anything under ``src/``. It exists to answer one question with a
picture instead of a table: **does each coarse window actually contain the
transition it is supposed to?**

That question decides the order of work. Pass 2 searches *inside* a window, so a
window that misses the truth cannot be rescued by refining it -- and a refined
answer from a wrong window is worse than no answer, because it looks precise.

    PYTHONPATH=src python scripts/show_windows.py outputs/track/dual

Pass ``--labels eval/labels.json`` to draw the hand-made ground truth alongside,
which is the only way to see whether a window contains the truth or merely looks
plausible. That option reads the labels, which is why this script lives outside
``src/`` -- the pipeline may never do so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from excavator_cycles.features import load as load_features  # noqa: E402
from excavator_cycles.fsm import calibrate, walk  # noqa: E402
from excavator_cycles.render import render  # noqa: E402

ONSET_KEYS = {
    "digging": "digging_begins",
    "hauling": "hauling_begins",
    "dumping": "dumping_begins",
    "swinging": "swinging_begins",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("track_dir", type=Path, help="a directory produced by `track`")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--hold-samples", type=int, default=3)
    parser.add_argument("--lookback-samples", type=int, default=8)
    parser.add_argument(
        "--labels",
        type=Path,
        default=REPO / "eval" / "labels.json",
        help="drawn dashed alongside the pipeline's own answer. Evaluation only.",
    )
    parser.add_argument("--no-labels", action="store_true", help="hide the ground truth")
    parser.add_argument("--no-video", action="store_true", help="print only; skip rendering")
    args = parser.parse_args(argv)

    table, _scene = load_features(args.track_dir)
    levels = calibrate(table)
    print(levels.report())

    found = walk(
        table,
        levels,
        hold_samples=args.hold_samples,
        lookback_samples=args.lookback_samples,
    )
    times = table.time_seconds
    windows = [
        (
            d.phase,
            float(times[d.window.lo]),
            float(times[min(d.window.hi, len(times) - 1)]),
        )
        for d in found
    ]

    truth = {}
    if args.labels and not args.no_labels and args.labels.exists():
        labels = json.loads(args.labels.read_text())
        fps = float(labels["video"]["fps"])
        truth = {k: labels["boundaries"][v] / fps for k, v in ONSET_KEYS.items()}

    fired = {}
    print(f"\n{len(found)} detections")
    print(
        f"  {'phase':10}{'window (s)':>20}{'width':>7}{'FIRED':>8}"
        f"{'truth':>8}{'err':>8}   in window?"
    )
    for (phase, start, end), detection in zip(windows, found, strict=True):
        when = float(times[detection.fired_at])
        fired.setdefault(phase, when)
        reference = truth.get(phase)
        verdict = error = ""
        if reference is not None:
            verdict = "yes" if start <= reference <= end else "NO"
            error = f"{when - reference:+.2f}"
        flag = "  out-of-seq" if detection.out_of_sequence else ""
        print(
            f"  {phase:10}{f'[{start:.2f}, {end:.2f}]':>20}{end - start:>7.2f}"
            f"{when:>8.2f}{reference or 0:>8.2f}{error:>8}   {verdict}{flag}"
        )

    if args.no_video:
        return 0

    out = args.out or args.track_dir / "pass1_windows.mp4"
    stats = render(
        args.track_dir,
        out_path=out,
        scale=args.scale,
        windows=windows,
        onsets=fired,
        reference=truth or None,
    )
    print(f"\n  wrote {stats.output_path}  ({stats.frames_written} frames)")
    print("    shaded span   the window pass 1 searched")
    print("    SOLID line    where the trigger actually fired")
    if truth:
        print("    dashed grey   the hand-labelled truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
