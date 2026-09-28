"""The shape cues on the 83 s clip: three cycles, hand-labelled.

`test_onset_accuracy.py` gates the dev clip, which holds one cycle. This gates the
clip the cues were chosen on, where a cue has to work three times in a row and an
early haul can cascade through the rest of its cycle. It also checks what the dev
clip cannot: that mirrored footage reads the same.

Like `test_onset_accuracy.py` it reads labels from `eval/`, which a test may do
and the pipeline may not.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.features import FeatureTable
from excavator_cycles.fsm import (
    STATES,
    calibrate,
    default_interrupts,
    locate,
    walk,
)

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "tests" / "fixtures" / "long_clip" / "features.npz"
LABELS = REPO / "eval" / "labels_long_clip.json"
TOLERANCE = 0.6

# Refined onset error per labelled onset, measured 2026-09-26 with the shape cues.
# The level cues found 7 onsets here, none within tolerance, and never a dump or a
# swing. Tighten as the cues improve. Hauling is early in two cycles (the bucket
# climbs while still scooping); the third dig is 0.02 s outside tolerance.
BASELINE_ABS_ERROR = [
    0.53,
    1.66,
    0.19,
    0.19,
    0.26,
    0.46,
    0.45,
    0.22,
    0.62,
    1.35,
    0.34,
    0.27,
    0.04,
]
SLACK = 0.05


def _table(path: Path = FEATURES) -> FeatureTable:
    z = np.load(path)
    return FeatureTable(**{f.name: z[f.name] for f in dataclasses.fields(FeatureTable)})


def _labelled() -> list[tuple[str, float]]:
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    fps = float(data["video"]["fps"])
    return [(o["phase"], o["frame"] / fps) for o in data["onsets"]["list"]]


def _onsets(table):
    config = Config()
    return locate(walk(table, calibrate(table, config), config=config), table, config)


@pytest.fixture(scope="module")
def onsets():
    return _onsets(_table())


def test_every_labelled_onset_is_found_in_order(onsets):
    assert [o.phase for o in onsets] == [p for p, _ in _labelled()]
    assert not any(o.out_of_sequence for o in onsets), "no cycle may be abandoned"


def test_onset_errors_do_not_regress(onsets):
    worse = []
    for i, (onset, (phase, truth)) in enumerate(zip(onsets, _labelled(), strict=True)):
        assert onset.refined is not None, f"{phase} at {truth:.2f} s was not measured"
        error = onset.refined - truth
        if abs(error) > BASELINE_ABS_ERROR[i] + SLACK:
            worse.append(
                f"#{i} {phase} at {truth:.2f} s: {error:+.2f} s, was {BASELINE_ABS_ERROR[i]}"
            )
    assert not worse, "these got worse than the recorded baseline:\n  " + "\n  ".join(worse)


def test_most_onsets_are_within_the_grading_tolerance(onsets):
    within = sum(
        abs(o.refined - truth) <= TOLERANCE
        for o, (_, truth) in zip(onsets, _labelled(), strict=True)
    )
    assert within >= 10, f"{within}/13 within {TOLERANCE} s"


def test_mirrored_footage_reads_the_same():
    """Filmed from the other side, every horizontal sign flips. The truck side is
    read from the video, and the swing cue is signed toward the truck, so the
    onsets must not move. Raw d2x/dt2 would have turned the swing's peak into a dip
    and lost every swing."""
    table = _table()
    mirrored = dataclasses.replace(
        table,
        dx_dt=-table.dx_dt,
        bucket_x=-table.bucket_x,
        rel_cabin_x=-table.rel_cabin_x,
        rel_truck_x=-table.rel_truck_x,
    )
    config = Config()
    assert calibrate(table, config).dump_side == -calibrate(mirrored, config).dump_side
    original, flipped = _onsets(table), _onsets(mirrored)
    assert [(o.phase, o.coarse) for o in original] == [(o.phase, o.coarse) for o in flipped]


def test_the_out_of_sequence_alarm_reads_the_digging_state_not_the_shape():
    """The dig onset's shape -- speed levelling off -- happens several times a
    cycle; used as the alarm, each would abandon the cycle in progress."""
    table = _table()
    levels = calibrate(table, Config())
    for i in range(0, len(table.time_seconds), 5):
        assert default_interrupts("digging", table, i, levels) == STATES["digging"](
            table, i, levels
        )
