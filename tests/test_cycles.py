"""Cycles: what the machine DID, and what we could MEASURE.

Two different questions, deliberately kept apart. A cycle the machine performed
is complete whether or not all four of its onsets could be pinned, so
`cycle_count` counts occurrences while the averages come from measurements. The
consequence -- that `cycle_count * average_cycle_duration` will NOT equal the
elapsed time -- is correct and will look like a bug to anyone reading it cold.
"""

from __future__ import annotations

import json

import pytest

from excavator_cycles.cycles import Answer, Cycle, assemble, summarise, write_answer

PHASES = ("digging", "hauling", "dumping", "swinging")


def _onsets(start=0.0, dig=2.0, haul=4.0, dump=6.0, swing=8.0):
    return {
        "digging": start + dig,
        "hauling": start + haul,
        "dumping": start + dump,
        "swinging": start + swing,
    }


# --- what makes a cycle complete ------------------------------------------


def test_a_cycle_with_all_four_onsets_is_measured():
    cycle = Cycle(onsets=_onsets(), ends=12.0, occurred=set(PHASES))
    assert cycle.complete and cycle.measurable
    assert cycle.reason is None


def test_a_missing_onset_WITH_evidence_counts_but_does_not_average():
    """Case 2. The phase happened -- the bucket was over the bed at some point
    -- and the cue failed to say when. A cycle occurred, so it counts; we cannot
    measure it, so it must not pollute the averages."""
    onsets = _onsets()
    del onsets["dumping"]
    cycle = Cycle(onsets=onsets, ends=12.0, occurred=set(PHASES))
    assert cycle.complete, "the machine did perform a full cycle"
    assert not cycle.measurable, "but we cannot time it"
    assert "dumping" in cycle.reason


def test_a_missing_onset_with_NO_evidence_is_not_a_cycle_at_all():
    """Case 3. The bucket never went near the truck: the machine re-dug instead
    of dumping, so no cycle happened and nothing should count it."""
    onsets = _onsets()
    del onsets["dumping"]
    cycle = Cycle(onsets=onsets, ends=12.0, occurred={"digging", "hauling", "swinging"})
    assert not cycle.complete
    assert not cycle.measurable
    assert "dumping" in cycle.reason


def test_onsets_out_of_order_are_not_measurable():
    onsets = _onsets()
    onsets["dumping"] = onsets["hauling"] - 0.5
    cycle = Cycle(onsets=onsets, ends=12.0, occurred=set(PHASES))
    assert not cycle.measurable
    assert "order" in cycle.reason.lower()


def test_a_cycle_that_never_closed_is_not_measurable():
    """No next digging onset means the clip ended mid-cycle."""
    cycle = Cycle(onsets=_onsets(), ends=None, occurred=set(PHASES))
    assert not cycle.measurable


# --- durations ------------------------------------------------------------


def test_phase_durations_are_the_gaps_between_onsets():
    cycle = Cycle(
        onsets=_onsets(dig=2, haul=4, dump=6, swing=8), ends=12.0, occurred=set(PHASES)
    )
    d = cycle.durations()
    assert d == {"digging": 2.0, "hauling": 2.0, "dumping": 2.0, "swinging": 4.0}


def test_swinging_runs_to_the_NEXT_digging_onset():
    """The spec: swinging ends immediately before the next digging phase."""
    cycle = Cycle(onsets=_onsets(swing=8), ends=15.0, occurred=set(PHASES))
    assert cycle.durations()["swinging"] == pytest.approx(7.0)


def test_cycle_duration_is_onset_to_next_onset():
    cycle = Cycle(onsets=_onsets(dig=2), ends=15.0, occurred=set(PHASES))
    assert cycle.duration == pytest.approx(13.0)


# --- assembling a run of detections ---------------------------------------


def test_head_and_tail_partials_fall_outside_every_cycle():
    """Not a special case: a cycle is dig-onset to dig-onset, so footage before
    the first and after the last is in no cycle at all."""
    onsets = [
        ("digging", 2.0),
        ("hauling", 4.0),
        ("dumping", 6.0),
        ("swinging", 8.0),
        ("digging", 12.0),
    ]
    cycles = assemble(onsets, occurred=set(PHASES))
    assert len(cycles) == 1
    assert cycles[0].onsets["digging"] == 2.0 and cycles[0].ends == 12.0


def test_three_cycles_assemble_into_three():
    onsets = []
    for i in range(4):  # four digging onsets bound THREE cycles
        base = i * 10.0
        onsets.append(("digging", base + 2))
        if i < 3:
            onsets += [("hauling", base + 4), ("dumping", base + 6), ("swinging", base + 8)]
    cycles = assemble(onsets, occurred=set(PHASES))
    assert len(cycles) == 3
    assert all(c.measurable for c in cycles)


def test_a_single_digging_onset_bounds_no_cycles():
    assert assemble([("digging", 2.0)], occurred=set(PHASES)) == []


# --- the answer -----------------------------------------------------------


def test_averages_come_only_from_measurable_cycles():
    good = Cycle(onsets=_onsets(), ends=12.0, occurred=set(PHASES))
    broken = dict(_onsets(start=20.0))
    del broken["dumping"]
    unmeasurable = Cycle(onsets=broken, ends=32.0, occurred=set(PHASES))
    answer = summarise([good, unmeasurable])
    assert answer.cycle_count == 2, "both cycles OCCURRED"
    assert answer.average_phase_duration_seconds["digging"] == pytest.approx(2.0), (
        "only the measurable one contributes to the average"
    )


def test_the_schema_matches_the_task_exactly(tmp_path):
    answer = summarise([Cycle(onsets=_onsets(), ends=12.0, occurred=set(PHASES))])
    path = write_answer(answer, tmp_path / "answer.json")
    data = json.loads(path.read_text())
    assert set(data) == {
        "cycle_count",
        "average_cycle_duration_seconds",
        "average_phase_duration_seconds",
    }
    assert set(data["average_phase_duration_seconds"]) == set(PHASES)
    assert isinstance(data["cycle_count"], int)


def test_no_measurable_cycles_is_reported_not_crashed():
    """A legitimate outcome on a clip too short to contain a full cycle."""
    answer = summarise([])
    assert answer.cycle_count == 0
    assert isinstance(answer, Answer)
