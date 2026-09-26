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

from excavator_cycles.config import Config  # noqa: E402
from excavator_cycles.features import load as load_features  # noqa: E402
from excavator_cycles.fsm import calibrate, samples_for, walk  # noqa: E402
from excavator_cycles.render import render  # noqa: E402

# Truth is paired BY POSITION, not by phase name. The cycle is
# dig -> haul -> dump -> swing -> dig, so the fifth transition is digging again
# and its truth is `cycle_ends`. Keying by name scored the closing dig against
# the opening one and printed a meaningless +23.95 s error. The dev clip holds
# one complete cycle, so anything past the fifth detection has no truth to be
# compared against and is shown without an error.
ONSET_SEQUENCE = (
    ("digging", "digging_begins"),
    ("hauling", "hauling_begins"),
    ("dumping", "dumping_begins"),
    ("swinging", "swinging_begins"),
    ("digging", "cycle_ends"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("track_dir", type=Path, help="a directory produced by `track`")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=2.0)
    defaults = Config().fsm
    parser.add_argument("--hold-seconds", type=float, default=defaults.hold_seconds)
    parser.add_argument("--lookback-seconds", type=float, default=defaults.lookback_seconds)
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
    levels = calibrate(table, Config())
    print(levels.report())

    times = table.time_seconds
    found = walk(
        table,
        levels,
        hold_samples=samples_for(args.hold_seconds, times),
        lookback_samples=samples_for(args.lookback_seconds, times),
    )
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
        truth = [
            (phase, float(labels["boundaries"][key]) / fps) for phase, key in ONSET_SEQUENCE
        ]

    fired = {}
    print(f"\n{len(found)} detections")
    print(
        f"  {'phase':10}{'window (s)':>20}{'width':>7}{'FIRED':>8}"
        f"{'truth':>8}{'err':>8}   in window?"
    )
    for index, ((phase, start, end), detection) in enumerate(zip(windows, found, strict=True)):
        when = float(times[detection.fired_at])
        # Every onset, not just the first per phase: `setdefault` silently dropped
        # the cycle-closing dig from the rendered overlay.
        fired.setdefault(phase, []).append(when)
        expected_phase, reference = truth[index] if index < len(truth) else (phase, None)
        verdict = error = ""
        if reference is not None:
            verdict = "yes" if start <= reference <= end else "NO"
            error = f"{when - reference:+.2f}"
            if expected_phase != phase:
                verdict += f"  (expected {expected_phase} here)"
        flag = "  out-of-seq" if detection.out_of_sequence else ""
        print(
            f"  {phase:10}{f'[{start:.2f}, {end:.2f}]':>20}{end - start:>7.2f}"
            f"{when:>8.2f}{reference if reference is not None else 0:>8.2f}{error:>8}"
            f"   {verdict}{flag}"
        )

    if args.no_video:
        return 0

    out = args.out or args.track_dir / "pass1_windows.mp4"
    stats = render(
        args.track_dir,
        out_path=out,
        scale=args.scale,
        windows=windows,
        # `render`'s overlay is keyed by phase name, so it can hold one onset per
        # phase and no more. The printed table above shows every onset; the video
        # shows the first of each. Drawing repeats needs `render` to take a list,
        # which is a change to the pipeline's API and not this script's business.
        onsets={phase: whens[0] for phase, whens in fired.items()},
        reference=dict(truth) or None,
    )
    print(f"\n  wrote {stats.output_path}  ({stats.frames_written} frames)")
    print("    shaded span   the window pass 1 searched")
    print("    SOLID line    where the trigger actually fired")
    if truth:
        print("    dashed grey   the hand-labelled truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
