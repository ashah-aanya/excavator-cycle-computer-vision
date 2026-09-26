#!/usr/bin/env python3
"""Render the reference video: the features, marked with the HAND LABELS.

EVALUATION ONLY. This script reads ``eval/labels.json``, which the pipeline is
forbidden to touch -- the task spec bans reading the answer. It lives in
``eval/`` for that reason, and nothing under ``src/excavator_cycles/`` imports
it. ``tests/test_eval_labels.py`` enforces that by grepping the package.

What it is for
--------------
The state machine does not exist yet, so there is nothing to compare against
except a person's eye. This produces the video that shows where the boundaries
*actually* are, drawn on the same six signals the state machine will read, so
the cue design can be argued from a picture rather than from a table of numbers.

When the state machine does exist, render the same clip twice -- once with these
labels and once with its predictions -- and the difference is the error, visible
rather than tabulated.

Usage
-----
    uv run python eval/annotate_solution.py outputs/track/dual
    uv run python eval/annotate_solution.py outputs/track/dual --out my_reference.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from excavator_cycles.render import render  # noqa: E402
from excavator_cycles.track import load_result  # noqa: E402

LABELS = Path(__file__).resolve().parent / "labels.json"

# labels.json names the moments; the pipeline names the phases they start.
ONSET_KEYS = {
    "digging": "digging_begins",
    "hauling": "hauling_begins",
    "dumping": "dumping_begins",
    "swinging": "swinging_begins",
}


def onset_seconds(labels: dict, fps: float) -> dict[str, float]:
    """The four hand-labelled onsets, in seconds.

    Boundaries are recorded as an instant *between* frames -- boundary ``f``
    means "immediately before frame f" -- so ``f / fps`` is the right conversion
    and no half-frame offset belongs here.
    """
    boundaries = labels["boundaries"]
    return {name: boundaries[key] / fps for name, key in ONSET_KEYS.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="annotate_solution.py",
        description=(
            "Render the reference video, marking the hand-labelled phase "
            "boundaries on the feature graphs. Evaluation only -- the pipeline "
            "must never read these labels."
        ),
    )
    parser.add_argument("track_dir", type=Path, help="a directory produced by `track`")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="destination (default: <track_dir>/annotated_sol.mp4)",
    )
    parser.add_argument("--scale", type=float, default=2.0, help="resize factor")
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args(argv)

    labels = json.loads(args.labels.read_text())
    result, _masks = load_result(args.track_dir)

    # The labels carry their own frame rate, measured rather than assumed. Using
    # the clip's own would silently rescale every boundary if the two disagreed,
    # so they are checked against each other instead.
    label_fps = float(labels["video"]["fps"])
    if abs(label_fps - result.fps) > 0.01:
        print(
            f"  WARNING: labels were made at {label_fps:.4f} fps, this run "
            f"reports {result.fps:.4f}. Using the labels' rate.",
            file=sys.stderr,
        )

    onsets = onset_seconds(labels, label_fps)
    out = args.out or args.track_dir / "annotated_sol.mp4"

    print(f"  labels : {args.labels}")
    print(f"  source : {labels['video']['filename']}")
    print("  hand-labelled onsets:")
    for name, when in onsets.items():
        frame = labels["boundaries"][ONSET_KEYS[name]]
        print(f"    {name:9s} {when:6.2f}s   (frame {frame})")
    cycle_end = labels["boundaries"]["cycle_ends"] / label_fps
    end_frame = labels["boundaries"]["cycle_ends"]
    print(f"    {'cycle end':9s} {cycle_end:6.2f}s   (frame {end_frame})")

    stats = render(args.track_dir, out_path=out, scale=args.scale, onsets=onsets)
    print()
    print(f"  wrote {stats.output_path}  ({stats.frames_written} frames)")
    print("  THIS IS THE REFERENCE, NOT A PREDICTION: the markers are hand labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
