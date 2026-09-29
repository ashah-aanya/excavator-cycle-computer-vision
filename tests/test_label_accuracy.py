"""The phase search against the two hand-labelled clips: a ratchet.

**Development only; not part of the submission.** These tests read the hand labels under
`eval/`, which a test may do and the pipeline may not (`test_eval_labels.py` fails if
anything under `src/` names them). The zip leaves this file out.

Both clips were used while the rules were designed, so the numbers are optimistic. They
are recorded so that a change to the rules cannot quietly make any labelled start worse:
each error may not grow past its recorded value by more than `SLACK`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from excavator_cycles.features import load
from excavator_cycles.starts import find_phase_starts

REPO = Path(__file__).resolve().parents[1]
PHASES = ("digging", "hauling", "dumping", "swinging")
TOLERANCE = 0.6  # the task's grading tolerance, seconds
SLACK = 0.05

CLIPS = {
    "dev_clip": REPO / "eval" / "labels.json",
    "long_clip": REPO / "eval" / "labels_long_clip.json",
}

# |start - label| per labelled onset, seconds, measured 2026-09-29 on the ported code.
BASELINE_ABS_ERROR = {
    # The dev clip's dump label (18.65 s) marks the bucket ARRIVING over the truck. The
    # search follows TIPPING (the aspect ratio falling), which the frames show starting
    # about 2 s later -- so 1.97 s is the definitions differing, not a mis-detection.
    "dev_clip": [0.63, 0.23, 1.97, 0.23, 0.17],
    "long_clip": [
        0.47,
        0.34,
        0.61,
        0.79,
        0.34,
        0.46,
        0.35,
        0.02,
        0.02,
        0.75,
        0.14,
        0.33,
        0.56,
    ],
}
INSIDE_ITS_WINDOW = {"dev_clip": 4, "long_clip": 13}
WITHIN_TOLERANCE = {"dev_clip": 3, "long_clip": 10}


def _labelled(path: Path) -> list[tuple[str, float]]:
    """Every labelled onset as (phase, seconds), in time order. Reads both label layouts:
    a list of onsets, or one cycle's named boundaries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    fps = float(data["video"]["fps"])
    if "onsets" in data:
        pairs = [(o["phase"], o["frame"] / fps) for o in data["onsets"]["list"]]
    else:
        b = data["boundaries"]
        pairs = [(p, b[f"{p}_begins"] / fps) for p in PHASES] + [
            ("digging", b["cycle_ends"] / fps)
        ]
    return sorted(pairs, key=lambda pair: pair[1])


@pytest.fixture(scope="module")
def searches():
    fixtures = REPO / "tests" / "fixtures"
    return {clip: find_phase_starts(*load(fixtures / clip)) for clip in CLIPS}


@pytest.mark.parametrize("clip", CLIPS)
def test_every_labelled_phase_is_found_in_order(searches, clip):
    found = [s.phase for s in searches[clip].starts]
    assert found == [phase for phase, _ in _labelled(CLIPS[clip])]


@pytest.mark.parametrize("clip", CLIPS)
def test_start_errors_do_not_regress(searches, clip):
    labelled = _labelled(CLIPS[clip])
    worse = []
    for i, (start, (phase, truth)) in enumerate(
        zip(searches[clip].starts, labelled, strict=True)
    ):
        error = start.time - truth
        if abs(error) > BASELINE_ABS_ERROR[clip][i] + SLACK:
            worse.append(
                f"#{i} {phase} at {truth:.2f} s: {error:+.2f} s, was {BASELINE_ABS_ERROR[clip][i]}"
            )
    assert not worse, "these got worse than the recorded baseline:\n  " + "\n  ".join(worse)


@pytest.mark.parametrize("clip", CLIPS)
def test_enough_starts_are_within_the_grading_tolerance(searches, clip):
    labelled = _labelled(CLIPS[clip])
    within = sum(
        abs(s.time - truth) <= TOLERANCE
        for s, (_, truth) in zip(searches[clip].starts, labelled, strict=True)
    )
    assert within >= WITHIN_TOLERANCE[clip], f"{within}/{len(labelled)} within {TOLERANCE} s"


@pytest.mark.parametrize("clip", CLIPS)
def test_the_windows_contain_the_labelled_onsets(searches, clip):
    labelled = _labelled(CLIPS[clip])
    inside = sum(
        s.window[0] <= truth <= s.window[1]
        for s, (_, truth) in zip(searches[clip].starts, labelled, strict=True)
    )
    assert inside >= INSIDE_ITS_WINDOW[clip], f"{inside}/{len(labelled)} inside their window"
