"""Does pass 1 actually bracket the transition it claims to? Measured, not assumed.

Why this file exists
--------------------
Every other test of the state machine is synthetic. ``_scripted(schedule)`` in
``test_fsm.py`` hands ``walk`` a table built to fire on cue, which is the right
way to verify *sequencing* -- phases advance in order, windows never overlap, an
out-of-sequence dig abandons the cycle -- and is structurally incapable of
verifying *accuracy*. A schedule invented by the test has no truth to be wrong
about, so a suite made only of them passes whether the cues work or not.

The cost was not hypothetical. All five pass-1 windows excluded the labelled
onset they existed to bracket, by -1.9 s to +4.8 s against a +/-0.6 s tolerance,
and 52 green tests could not fail on it. ``scripts/show_windows.py`` printed
``in window? NO`` five times; nothing turned that into a build failure.

The two-part shape, and why
---------------------------
``test_every_window_contains_its_onset`` is the requirement. It is the thing
pass 2 needs to be true, because pass 2 searches INSIDE the window: an onset
outside it is unreachable, and refining a wrong window is worse than not
refining, because the answer comes back looking precise. It is marked
``xfail(strict=True)``, which means the build fails BOTH ways -- while the gap
persists it stays quiet, and the moment a cue fix makes it pass, CI breaks and
forces someone to delete the marker. An xfail that silently starts passing is
how a fixed bug gets re-broken later.

``test_onset_errors_do_not_regress`` is the ratchet. It pins today's measured
error per phase so cue work can be judged: any change that makes a phase worse
than its recorded baseline fails here, immediately, with the number. Without it
"fix the cues" is four weeks of guessing.

These read ``eval/labels.json``. That is allowed for a test and forbidden for
the pipeline -- ``test_pipeline_never_references_the_labels`` enforces the half
that matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.features import load as load_features
from excavator_cycles.fsm import calibrate, samples_for, walk

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dev_clip"
LABELS = REPO / "eval" / "labels.json"

# The cycle runs dig -> haul -> dump -> swing -> dig, so the FIFTH transition is
# a digging onset again and its truth is `cycle_ends`, not `digging_begins`.
# Keying truth by phase name instead of by position is a real bug that was in
# `scripts/show_windows.py`: it scored the closing dig against the opening one
# and reported a meaningless +23.95 s.
EXPECTED_SEQUENCE = (
    ("digging", "digging_begins"),
    ("hauling", "hauling_begins"),
    ("dumping", "dumping_begins"),
    ("swinging", "swinging_begins"),
    ("digging", "cycle_ends"),
)

# Measured on this fixture, 2026-09-26, and every one of them is a defect. These
# are a floor to work up from, not a target: the grading tolerance is 0.6 s, so
# only the last of these five would score at all. Tighten each as cues improve.
BASELINE_ABS_ERROR = {
    "digging_begins": 0.77,
    "hauling_begins": 1.97,
    "dumping_begins": 5.14,
    "swinging_begins": 2.67,
    "cycle_ends": 1.23,
}
SLACK = 0.05  # absorbs float noise in the timestamps; far below the 0.6 s budget


@pytest.fixture(scope="module")
def detections():
    """`calibrate` + `walk` on a real feature table, exactly as the CLI runs them."""
    table, _scene = load_features(FIXTURE)
    config = Config()
    times = table.time_seconds
    found = walk(
        table,
        calibrate(table, config),
        hold_samples=samples_for(config.fsm.hold_seconds, times),
        lookback_samples=samples_for(config.fsm.lookback_seconds, times),
    )
    return table, found


@pytest.fixture(scope="module")
def truth() -> dict[str, float]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    fps = float(labels["video"]["fps"])
    boundaries = labels["boundaries"]
    # Only the five keys this file names. `boundaries` also carries `_`-prefixed
    # prose (`_units`, `_definition_source`), and taking "everything that is an
    # int" would silently absorb any new key added later.
    wanted = {key for _, key in EXPECTED_SEQUENCE}
    missing = wanted - boundaries.keys()
    assert not missing, f"eval/labels.json lost a boundary this gate needs: {missing}"
    return {key: float(boundaries[key]) / fps for key in wanted}


def test_the_walk_finds_the_five_transitions_in_order(detections):
    """The precondition for everything below: without this the pairing is a lie.

    Asserted separately and first, because a comparison of window to truth is
    meaningless if the Nth detection is not the Nth transition. It also is not
    vacuous the way `not any(...)` over a possibly-empty list would be.
    """
    _table, found = detections
    assert [d.phase for d in found] == [phase for phase, _ in EXPECTED_SEQUENCE]
    assert not any(d.out_of_sequence for d in found)


def test_onset_errors_do_not_regress(detections, truth):
    """The ratchet. Signed error reported, absolute error asserted."""
    table, found = detections
    times = table.time_seconds
    worse = []
    for detection, (phase, key) in zip(found, EXPECTED_SEQUENCE, strict=True):
        error = float(times[detection.fired_at]) - truth[key]
        limit = BASELINE_ABS_ERROR[key]
        if abs(error) > limit + SLACK:
            worse.append(f"{phase} ({key}): {error:+.2f} s, baseline was {limit:.2f} s")
    assert not worse, "these got worse than the recorded baseline:\n  " + "\n  ".join(worse)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "All five windows miss their onset. Pass 2 searches inside the window, so "
        "each of these is currently unreachable. Fixing the cues is what turns this "
        "green -- and when it does, this marker must be deleted, which is what "
        "strict=True is for."
    ),
)
def test_every_window_contains_its_onset(detections, truth):
    """The requirement, stated as the build will eventually enforce it."""
    table, found = detections
    times = table.time_seconds
    missed = []
    for detection, (phase, key) in zip(found, EXPECTED_SEQUENCE, strict=True):
        lo = float(times[detection.window.lo])
        hi = float(times[min(detection.window.hi, len(times) - 1)])
        target = truth[key]
        if not lo <= target <= hi:
            gap = lo - target if target < lo else target - hi
            missed.append(
                f"{phase} ({key}): [{lo:.2f}, {hi:.2f}] misses {target:.2f} by {gap:.2f}s"
            )
    assert not missed, "windows that cannot contain their onset:\n  " + "\n  ".join(missed)


def test_the_fixture_is_the_clip_the_labels_describe(truth):
    """Guards the pairing itself: a regenerated fixture from another video would
    make every number above meaningless while still passing."""
    table, _scene = load_features(FIXTURE)
    times = table.time_seconds
    assert len(times) == 296
    assert float(times[0]) == pytest.approx(0.0, abs=0.11)
    # The clip is 886 frames at 29.974 fps = 29.56 s; the last sample sits within
    # one sampling interval of that.
    assert float(times[-1]) == pytest.approx(29.56, abs=0.11)
    assert np.all(np.diff(times) > 0), "timestamps must be strictly increasing"
    assert truth["cycle_ends"] < float(times[-1])
