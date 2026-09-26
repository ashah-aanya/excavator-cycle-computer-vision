"""The two populations must stay separate all the way down.

`cycles.py` promises that `cycle_count` counts what HAPPENED while the averages
come from what could be MEASURED. That promise was defeated upstream, twice, and
both failures showed up as a single wrong number in `answer.json`.

**A failed refinement removed the cycle from both populations.** `locate` dropped
any detection whose pass-2 cue found nothing, reasoning that a made-up onset
would be indistinguishable from a measured one. True, and fine for the averages
-- but a dropped DIGGING onset is a dropped cycle BOUNDARY, and `assemble` splits
on digging. On the dev clip both digging refinements failed, so a video holding
one complete, fully-detected cycle reported `cycle_count: 0`. The cue being
imprecise is a fact about the pipeline; the cycle still happened.

**Evidence was global, not per-cycle.** `cli` passed
`occurred={d.phase for d in detections}` -- one set, built from the whole video,
copied into every cycle. Any video where some cycle somewhere had a dumping
detection marked EVERY cycle as having dumped. With 1.2 cycles in the dev clip
that is invisible; with four it silently inflates `cycle_count`.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.cycles import Cycle, assemble, summarise
from excavator_cycles.fsm import Onset

PHASES = ("digging", "hauling", "dumping", "swinging")


def cycle_of(
    start: float, step: float = 1.0, *, unrefined: tuple[str, ...] = ()
) -> list[Onset]:
    """Four onsets a second apart, with `unrefined` ones lacking a pass-2 time."""
    return [
        Onset(phase, None if phase in unrefined else start + i * step, start + i * step)
        for i, phase in enumerate(PHASES)
    ]


def test_a_failed_digging_refinement_still_counts_the_cycle():
    """The exact dev-clip failure: both digs unrefined, one real cycle, count 1.

    The averages must stay empty, because nothing was measured -- that is the
    honest pair of answers, and it is the one the spec's two questions ask for.
    """
    onsets = [*cycle_of(0.0, unrefined=("digging",)), Onset("digging", None, 10.0)]
    cycles = assemble(onsets)
    assert len(cycles) == 1
    assert cycles[0].complete, "all four phases were detected; the cycle happened"
    assert not cycles[0].measurable, "digging has no refined onset, so nothing to measure"
    answer = summarise(cycles)
    assert answer.cycle_count == 1
    assert answer.average_cycle_duration_seconds == 0.0


def test_a_fully_refined_cycle_is_both_counted_and_measured():
    """The control. Without this the test above passes on a broken `measurable`."""
    cycles = assemble([*cycle_of(0.0), Onset("digging", 10.0, 10.0)])
    assert len(cycles) == 1
    assert cycles[0].complete and cycles[0].measurable
    answer = summarise(cycles)
    assert answer.cycle_count == 1
    assert answer.average_cycle_duration_seconds == pytest.approx(10.0)
    assert answer.average_phase_duration_seconds["digging"] == pytest.approx(1.0)
    assert answer.average_phase_duration_seconds["swinging"] == pytest.approx(7.0)


def test_evidence_is_scoped_to_its_own_cycle():
    """Cycle 1 dumps, cycle 2 does not. Only cycle 1 may be complete."""
    onsets = [
        *cycle_of(0.0),  # dig, haul, dump, swing
        Onset("digging", 10.0, 10.0),
        Onset("hauling", 11.0, 11.0),
        Onset("swinging", 12.0, 12.0),  # no dumping this time round
        Onset("digging", 20.0, 20.0),
    ]
    cycles = assemble(onsets)
    assert len(cycles) == 2
    assert cycles[0].complete, "cycle 1 had all four phases"
    assert not cycles[1].complete, "cycle 2 never dumped -- a global set would hide this"
    assert "dumping" not in cycles[1].occurred
    assert summarise(cycles).cycle_count == 1


def test_evidence_can_come_from_a_weaker_check_than_an_onset():
    """The mechanism `cycles.py` documented and did not have.

    A phase whose onset was never located may still have left evidence inside the
    span. That makes the cycle COUNTED but not MEASURED -- the middle case, and
    the only reason the two populations are worth separating at all.
    """
    onsets = [
        Onset("digging", 0.0, 0.0),
        Onset("hauling", 1.0, 1.0),
        Onset("swinging", 3.0, 3.0),  # dumping's onset never located
        Onset("digging", 10.0, 10.0),
    ]
    # The span 0-10 s did contain evidence of dumping, even though no onset was
    # pinned: the bucket was over the bed at some point.
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    assert len(cycles) == 1
    assert cycles[0].complete, "evidence says it happened"
    assert not cycles[0].measurable, "no onset, so no duration"
    assert "occurred but its onset was not located" in cycles[0].reason
    assert summarise(cycles).cycle_count == 1


def test_a_span_with_no_evidence_of_a_phase_is_not_a_cycle():
    """The other side: a mid-video re-dig is not a complete cycle."""
    onsets = [
        Onset("digging", 0.0, 0.0),
        Onset("hauling", 1.0, 1.0),
        Onset("digging", 5.0, 5.0),
    ]
    cycles = assemble(onsets, evidence=lambda start, end: {"digging", "hauling"})
    assert len(cycles) == 1
    assert not cycles[0].complete
    assert "no cycle occurred" in cycles[0].reason
    assert summarise(cycles).cycle_count == 0


def test_the_span_is_always_available_even_when_nothing_refined():
    """Evidence is asked about a span, so the span cannot depend on refinement."""
    onsets = [
        Onset("digging", None, 2.0),
        Onset("hauling", None, 4.0),
        Onset("digging", None, 9.0),
    ]
    asked: list[tuple[float, float]] = []

    def evidence(start: float, end: float) -> set[str]:
        asked.append((start, end))
        return {"digging", "hauling"}

    cycles = assemble(onsets, evidence=evidence)
    assert asked == [(2.0, 9.0)], "the coarse trigger times bound the span"
    assert cycles[0].span == (2.0, 9.0)


def test_cycle_count_times_average_need_not_equal_elapsed_time():
    """Stated loudly in the module docstring, so it is worth pinning.

    One counted cycle that could not be measured, one that could: the count is 2
    and the average comes from a single cycle. Their product means nothing, and
    that is correct rather than a bug.
    """
    onsets = [
        *cycle_of(0.0, unrefined=("dumping",)),
        *cycle_of(10.0),
        Onset("digging", 20.0, 20.0),
    ]
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    answer = summarise(cycles)
    assert answer.cycle_count == 2
    assert sum(1 for c in cycles if c.measurable) == 1
    assert answer.average_cycle_duration_seconds == pytest.approx(10.0)


def test_an_unrefined_closing_onset_blocks_measurement_but_not_counting():
    """Swinging is measured against the NEXT dig, so that dig must be refined."""
    onsets = [*cycle_of(0.0), Onset("digging", None, 10.0)]
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    assert cycles[0].complete and not cycles[0].measurable
    assert summarise(cycles).cycle_count == 1


def test_onsets_out_of_order_are_reported_rather_than_averaged():
    """A negative duration must never reach the answer."""
    onsets = [
        Onset("digging", 0.0, 0.0),
        Onset("hauling", 5.0, 5.0),
        Onset("dumping", 2.0, 2.0),  # impossible: before hauling
        Onset("swinging", 6.0, 6.0),
        Onset("digging", 10.0, 10.0),
    ]
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    assert not cycles[0].measurable
    assert "out of order" in cycles[0].reason
    assert summarise(cycles).average_cycle_duration_seconds == 0.0


def test_head_and_tail_partials_fall_outside_every_cycle():
    """The spec's ignore-the-partials rule, which needs no special case."""
    onsets = [
        Onset("swinging", -1.0, -1.0),  # tail of a cycle that began before the clip
        *cycle_of(0.0),
        Onset("digging", 10.0, 10.0),
        Onset("hauling", 11.0, 11.0),  # head of a cycle the clip cuts off
    ]
    cycles = assemble(onsets)
    assert len(cycles) == 1, "two digging onsets bound exactly one cycle"
    assert cycles[0].span == (0.0, 10.0)


def test_a_cycle_needs_two_digging_onsets():
    """Nothing to bound, so nothing to report -- and not an exception."""
    assert assemble([Onset("digging", 0.0, 0.0), Onset("hauling", 1.0, 1.0)]) == []
    answer = summarise([])
    assert answer.cycle_count == 0
    assert answer.average_phase_duration_seconds == dict.fromkeys(PHASES, 0.0)


def test_cycle_is_still_constructible_directly_for_callers_that_have_onsets():
    """`Cycle` stays usable without `assemble`, which the CLI's breakdown needs."""
    cycle = Cycle(
        onsets=dict.fromkeys(PHASES, 0.0) | {"hauling": 1.0, "dumping": 2.0, "swinging": 3.0},
        ends=4.0,
        span=(0.0, 4.0),
        occurred=set(PHASES),
    )
    assert cycle.complete and cycle.measurable
    assert cycle.durations()["swinging"] == pytest.approx(1.0)
    assert np.isclose(cycle.duration, 4.0)


def test_an_interrupted_cycle_is_not_counted_as_a_complete_one():
    """An out-of-sequence dig abandons the cycle being built. Its already-emitted
    detections stay in the list, unmarked, so the guard has to be downstream.

    It is: the interrupting dig is itself a cycle boundary, so the abandoned span
    is bounded by two digs like any other and judged on its own evidence. A span
    that never dumped is not a complete cycle, whatever was detected inside it.
    That is why the evidence check has to be per-span -- a global set would have
    called this complete.
    """
    onsets = [
        Onset("digging", 0.0, 0.0),
        Onset("hauling", 1.0, 1.0),
        # the machine went back to the pile instead of dumping
        Onset("digging", 4.0, 4.0, out_of_sequence=True),
        *cycle_of(10.0)[1:],
        Onset("digging", 20.0, 20.0),
    ]
    cycles = assemble(
        onsets,
        evidence=lambda start, end: {"digging", "hauling"} if end <= 4.0 else set(PHASES),
    )
    assert len(cycles) == 2
    assert not cycles[0].complete, "no dumping happened in the abandoned span"
    assert "no dumping, swinging in this span" in cycles[0].reason
    assert cycles[1].complete
    assert summarise(cycles).cycle_count == 1


def test_evidence_within_really_is_restricted_to_the_span():
    """The claimed fix, tested with the REAL function rather than a lambda.

    Every other test here passes a fake `evidence` callable, so none of them touches
    the actual windowing -- replacing `(times >= start) & (times <= end)` with the
    whole video left the entire suite green. That is the exact bug the CLI change
    claims to have removed, so it needs a test that can see it.

    Two cycles: the first dumps, the second does not. A whole-video evidence check
    reports dumping in both and marks both complete; a per-span one does not.
    """
    import numpy as np

    from excavator_cycles.fsm import Levels, Split, evidence_within

    class Table:
        """Only the columns the four triggers read."""

        def __init__(self, n: int):
            self.time_seconds = np.arange(n) * 1.0
            self.found = np.ones(n, bool)
            self.height = np.zeros(n)
            self.dh_dt = np.zeros(n)
            self.d2h_dt2 = np.zeros(n)
            self.speed_x = np.zeros(n)
            self.truck_overlap = np.zeros(n)
            self.rel_cabin_x = np.full(n, 0.4)
            self.aspect_ratio = np.ones(n)

    table = Table(20)
    # Dumping needs overlap above the level AND the bucket past the cabin. Only
    # samples 4-6 qualify, which lie inside the FIRST span and no other.
    table.truck_overlap[4:7] = 0.8
    levels = Levels(
        low_height=Split(0.1, 0.95, 50, 50),
        over_truck=Split(0.2, 0.95, 50, 50),
        moving=Split(0.3, 0.95, 50, 50),
        dump_side=1.0,
    )

    first = evidence_within(table, levels, 0.0, 9.0)
    second = evidence_within(table, levels, 10.0, 19.0)
    whole = evidence_within(table, levels, 0.0, 19.0)

    assert "dumping" in first, "the dump is inside the first span"
    assert "dumping" not in second, (
        "the second span holds no dump -- a whole-video check would say otherwise"
    )
    assert "dumping" in whole, "and the whole video does contain one, which is the trap"
    assert first != second, "the two spans must not produce the same evidence"


def test_assemble_asks_the_evidence_function_for_each_span_separately():
    """The wiring, not the windowing: one call per cycle, with that cycle's bounds."""
    asked: list[tuple[float, float]] = []

    def evidence(start: float, end: float) -> set[str]:
        asked.append((start, end))
        return set(PHASES)

    onsets = [*cycle_of(0.0), *cycle_of(10.0), Onset("digging", 20.0, 20.0)]
    assemble(onsets, evidence=evidence)
    assert asked == [(0.0, 10.0), (10.0, 20.0)], f"expected one call per cycle, got {asked}"


def test_a_sample_on_a_cycle_boundary_belongs_to_exactly_one_cycle():
    """The evidence span is HALF-OPEN, matching `Window`.

    With both ends inclusive, the closing digging onset's sample belonged to this
    cycle AND was the next cycle's opening sample. Harmless for digging, which is the
    boundary itself -- but a single dumping sample sitting exactly on a boundary
    counted as evidence in both spans, which is enough to mark the wrong cycle
    complete and inflate `cycle_count`.
    """
    import numpy as np

    from excavator_cycles.fsm import Levels, Split, evidence_within

    class Table:
        def __init__(self, n: int):
            self.time_seconds = np.arange(n) * 1.0
            self.found = np.ones(n, bool)
            for column in ("height", "dh_dt", "d2h_dt2", "speed_x", "truck_overlap"):
                setattr(self, column, np.zeros(n))
            self.aspect_ratio = np.ones(n)
            self.rel_cabin_x = np.full(n, 0.4)

    table = Table(12)
    table.truck_overlap[5] = 0.8  # exactly one dumping sample, exactly on the boundary
    clean = Split(0.2, 0.95, 50, 50)
    levels = Levels(
        low_height=Split(0.1, 0.95, 50, 50),
        over_truck=clean,
        moving=Split(0.3, 0.95, 50, 50),
        dump_side=1.0,
        over_truck_observed=clean,
    )

    first = evidence_within(table, levels, 0.0, 5.0)
    second = evidence_within(table, levels, 5.0, 9.0)
    assert "dumping" not in first, "the boundary sample belongs to the NEXT span"
    assert "dumping" in second
    assert not ({"dumping"} & first & second), "no sample may be evidence in two cycles"
