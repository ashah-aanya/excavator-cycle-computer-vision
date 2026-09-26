#!/usr/bin/env python3
"""DIAGNOSTIC, not pipeline. Where did pass 1 think each transition happened?

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
        default=None,
        help="eval/labels.json, to draw the truth alongside. Evaluation only.",
    )
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
    if args.labels:
        labels = json.loads(args.labels.read_text())
        fps = float(labels["video"]["fps"])
        truth = {k: labels["boundaries"][v] / fps for k, v in ONSET_KEYS.items()}

    print(f"\n{len(found)} detections")
    print(f"  {'phase':10}{'window (s)':>20}{'width':>7}{'truth':>8}   contains it?")
    for (phase, start, end), detection in zip(found and windows, found, strict=True):
        reference = truth.get(phase)
        verdict = ""
        if reference is not None:
            verdict = "yes" if start <= reference <= end else "NO"
        flag = "  out-of-seq" if detection.out_of_sequence else ""
        print(
            f"  {phase:10}{f'[{start:.2f}, {end:.2f}]':>20}{end - start:>7.2f}"
            f"{reference if reference else 0:>8.2f}   {verdict}{flag}"
        )

    out = args.out or args.track_dir / "pass1_windows.mp4"
    stats = render(
        args.track_dir,
        out_path=out,
        scale=args.scale,
        windows=windows,
        onsets=truth or None,
    )
    print(f"\n  wrote {stats.output_path}  ({stats.frames_written} frames)")
    print("  shaded span = the window pass 1 chose")
    if truth:
        print("  dashed line = the hand-labelled truth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
