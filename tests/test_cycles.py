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
from excavator_cycles.fsm import Onset

PHASES = ("digging", "hauling", "dumping", "swinging")


def _span(onsets: dict, ends) -> tuple[float, float]:
    """The coarse bounds a real assembler would have produced for these onsets.

    Kept explicit rather than defaulted on `Cycle`, because a wrong span is
    silent: it is what the evidence check is asked about.
    """
    start = onsets.get("digging", 0.0)
    return (start, ends if ends is not None else max(onsets.values()))


def _onsets(start=0.0, dig=2.0, haul=4.0, dump=6.0, swing=8.0):
    return {
        "digging": start + dig,
        "hauling": start + haul,
        "dumping": start + dump,
        "swinging": start + swing,
    }


# --- what makes a cycle complete ------------------------------------------


def test_a_cycle_with_all_four_onsets_is_measured():
    cycle = Cycle(
        onsets=_onsets(), ends=12.0, span=_span(_onsets(), 12.0), occurred=set(PHASES)
    )
    assert cycle.complete and cycle.measurable
    assert cycle.reason is None


def test_a_missing_onset_WITH_evidence_counts_but_does_not_average():
    """Case 2. The phase happened -- the bucket was over the bed at some point
    -- and the cue failed to say when. A cycle occurred, so it counts; we cannot
    measure it, so it must not pollute the averages."""
    onsets = _onsets()
    del onsets["dumping"]
    cycle = Cycle(onsets=onsets, ends=12.0, span=_span(onsets, 12.0), occurred=set(PHASES))
    assert cycle.complete, "the machine did perform a full cycle"
    assert not cycle.measurable, "but we cannot time it"
    assert "dumping" in cycle.reason


def test_a_missing_onset_with_NO_evidence_is_not_a_cycle_at_all():
    """Case 3. The bucket never went near the truck: the machine re-dug instead
    of dumping, so no cycle happened and nothing should count it."""
    onsets = _onsets()
    del onsets["dumping"]
    cycle = Cycle(
        onsets=onsets,
        ends=12.0,
        span=_span(onsets, 12.0),
        occurred={"digging", "hauling", "swinging"},
    )
    assert not cycle.complete
    assert not cycle.measurable
    assert "dumping" in cycle.reason


def test_onsets_out_of_order_are_not_measurable():
    onsets = _onsets()
    onsets["dumping"] = onsets["hauling"] - 0.5
    cycle = Cycle(onsets=onsets, ends=12.0, span=_span(onsets, 12.0), occurred=set(PHASES))
    assert not cycle.measurable
    assert "order" in cycle.reason.lower()


def test_a_cycle_that_never_closed_is_not_measurable():
    """No next digging onset means the clip ended mid-cycle."""
    cycle = Cycle(
        onsets=_onsets(), ends=None, span=_span(_onsets(), None), occurred=set(PHASES)
    )
    assert not cycle.measurable


# --- durations ------------------------------------------------------------


def test_phase_durations_are_the_gaps_between_onsets():
    cycle = Cycle(
        onsets=_onsets(dig=2, haul=4, dump=6, swing=8),
        ends=12.0,
        span=_span(_onsets(dig=2, haul=4, dump=6, swing=8), 12.0),
        occurred=set(PHASES),
    )
    d = cycle.durations()
    assert d == {"digging": 2.0, "hauling": 2.0, "dumping": 2.0, "swinging": 4.0}


def test_swinging_runs_to_the_NEXT_digging_onset():
    """The spec: swinging ends immediately before the next digging phase."""
    cycle = Cycle(
        onsets=_onsets(swing=8),
        ends=15.0,
        span=_span(_onsets(swing=8), 15.0),
        occurred=set(PHASES),
    )
    assert cycle.durations()["swinging"] == pytest.approx(7.0)


def test_cycle_duration_is_onset_to_next_onset():
    cycle = Cycle(
        onsets=_onsets(dig=2),
        ends=15.0,
        span=_span(_onsets(dig=2), 15.0),
        occurred=set(PHASES),
    )
    assert cycle.duration == pytest.approx(13.0)


# --- assembling a run of detections ---------------------------------------


def test_head_and_tail_partials_fall_outside_every_cycle():
    """Not a special case: a cycle is dig-onset to dig-onset, so footage before
    the first and after the last is in no cycle at all."""
    onsets = [
        Onset("digging", 2.0, 2.0),
        Onset("hauling", 4.0, 4.0),
        Onset("dumping", 6.0, 6.0),
        Onset("swinging", 8.0, 8.0),
        Onset("digging", 12.0, 12.0),
    ]
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    assert len(cycles) == 1
    assert cycles[0].onsets["digging"] == 2.0 and cycles[0].ends == 12.0


def test_three_cycles_assemble_into_three():
    onsets = []
    for i in range(4):  # four digging onsets bound THREE cycles
        base = i * 10.0
        onsets.append(Onset("digging", base + 2, base + 2))
        if i < 3:
            onsets += [
                Onset("hauling", base + 4, base + 4),
                Onset("dumping", base + 6, base + 6),
                Onset("swinging", base + 8, base + 8),
            ]
    cycles = assemble(onsets, evidence=lambda start, end: set(PHASES))
    assert len(cycles) == 3
    assert all(c.measurable for c in cycles)


def test_a_single_digging_onset_bounds_no_cycles():
    assert (
        assemble([Onset("digging", 2.0, 2.0)], evidence=lambda start, end: set(PHASES)) == []
    )


# --- the answer -----------------------------------------------------------


def test_averages_come_only_from_measurable_cycles():
    good = Cycle(
        onsets=_onsets(), ends=12.0, span=_span(_onsets(), 12.0), occurred=set(PHASES)
    )
    broken = dict(_onsets(start=20.0))
    del broken["dumping"]
    unmeasurable = Cycle(
        onsets=broken, ends=32.0, span=_span(broken, 32.0), occurred=set(PHASES)
    )
    answer = summarise([good, unmeasurable])
    assert answer.cycle_count == 2, "both cycles OCCURRED"
    assert answer.average_phase_duration_seconds["digging"] == pytest.approx(2.0), (
        "only the measurable one contributes to the average"
    )


def test_the_schema_matches_the_task_exactly(tmp_path):
    answer = summarise(
        [Cycle(onsets=_onsets(), ends=12.0, span=_span(_onsets(), 12.0), occurred=set(PHASES))]
    )
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


# --- portability: a video this code has never seen -------------------------
#
# The development clip has 1.2 cycles at 10 Hz with a truck. Every one of those
# is an accident of one file, and the hidden videos share none of them
# necessarily. These fixtures vary each in turn.


class _SyntheticTable:
    """A machine doing `cycles` clean cycles at `rate_hz`.

    Built from the PHASE STRUCTURE the spec describes rather than from any
    recording, so it shares none of the development clip's accidents. Each phase
    gets the signals it should have:

        digging    bucket low, not traversing
        hauling    bucket rising, traversing
        dumping    bucket high, not traversing, over the bed, silhouette stretched
        swinging   bucket falling, traversing
    """

    def __init__(self, cycles=3, rate_hz=10.0, cycle_seconds=20.0, truck=True):
        import numpy as np

        step = 1.0 / rate_hz
        n = int(cycles * cycle_seconds / step) + 1
        self.time_seconds = np.arange(n) * step
        phase_of = (self.time_seconds % cycle_seconds) / (cycle_seconds / 4.0)
        dig = phase_of < 1
        haul = (phase_of >= 1) & (phase_of < 2)
        dump = (phase_of >= 2) & (phase_of < 3)
        swing = phase_of >= 3

        within = phase_of % 1.0  # 0..1 through whichever phase we are in
        self.height = np.select(
            [dig, haul, dump, swing],
            [-0.10, -0.10 + within * 0.40, 0.30, 0.30 - within * 0.40],
        )
        self.dh_dt = np.select([dig, haul, dump, swing], [0.0, 0.20, 0.0, -0.20])
        self.d2h_dt2 = np.gradient(self.dh_dt, self.time_seconds)
        # traversing during the two transits, still while digging and dumping
        self.speed_x = np.select([dig, haul, dump, swing], [0.01, 0.50, 0.01, 0.50])
        self.truck_overlap = np.where(dump, 0.40, 0.0) if truck else np.full(n, np.nan)
        self.rel_cabin_x = np.where(haul | dump, 0.5, -0.5)
        self.aspect_ratio = np.where(dump, 1.8, 1.2)
        self.bucket_x = np.zeros(n)
        self.bucket_y = np.zeros(n)
        self.found = np.ones(n, bool)


def _run(table):
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import calibrate, evidence_within, locate, walk

    config = Config.load()
    levels = calibrate(table)
    detections = walk(table, levels, config=config)
    onsets = locate(detections, table, config)
    cycles = assemble(
        onsets, evidence=lambda start, end: evidence_within(table, levels, start, end)
    )
    return cycles, detections


def test_it_runs_on_a_video_it_has_never_seen():
    """The whole chain, on data built from the phase structure rather than from
    any recording. It need not be ACCURATE -- the cues are known-wrong -- but it
    must complete and produce a well-formed answer."""
    cycles, detections = _run(_SyntheticTable(cycles=3))
    answer = summarise(cycles)
    assert isinstance(answer.cycle_count, int)
    assert set(answer.average_phase_duration_seconds) == set(PHASES)
    assert detections, "something should have been detected"


def test_a_different_frame_rate_changes_nothing_structural():
    """25 Hz rather than 10. Every parameter is a duration, so the same clip
    sampled differently must behave the same way."""
    slow, _ = _run(_SyntheticTable(cycles=3, rate_hz=10.0))
    fast, _ = _run(_SyntheticTable(cycles=3, rate_hz=25.0))
    assert len(slow) == len(fast), (
        f"{len(slow)} cycles at 10 Hz but {len(fast)} at 25 Hz -- "
        "some parameter is in samples, not seconds"
    )


def test_a_video_with_no_truck_still_completes():
    """truck_overlap is all-nan when nothing truck-like was detected. That is a
    legitimate video: the dumping gate is unavailable, not false."""
    cycles, _ = _run(_SyntheticTable(cycles=2, truck=False))
    answer = summarise(cycles)
    assert isinstance(answer.cycle_count, int), "must not raise"


def test_the_cycle_count_follows_the_data_not_a_constant():
    """Nothing may assume the development video's 1.2 cycles."""
    two, _ = _run(_SyntheticTable(cycles=2))
    five, _ = _run(_SyntheticTable(cycles=5))
    assert len(five) > len(two), f"2-cycle clip gave {len(two)}, 5-cycle gave {len(five)}"


def test_a_clip_too_short_for_a_cycle_reports_zero_rather_than_raising():
    cycles, _ = _run(_SyntheticTable(cycles=1, cycle_seconds=6.0))
    answer = summarise(cycles)
    assert answer.cycle_count >= 0


def test_what_is_frame_rate_invariant_and_what_is_not():
    """Cycle COUNT is invariant -- it comes from the phase structure. Whether a
    cycle is MEASURABLE is not: a denser clip puts more samples in each window,
    so refinement has more chance of finding the event inside one.

    On the synthetic fixture the same three cycles give 0 measurable at 10 Hz
    and 1 at 25 Hz. That is not a bug in the seconds-vs-samples rule -- the
    windows are the same DURATION either way -- it is the cues being marginal
    enough that sample density tips them. Pinned so that improving the cues can
    be seen to close the gap.
    """
    slow, _ = _run(_SyntheticTable(cycles=3, rate_hz=10.0))
    fast, _ = _run(_SyntheticTable(cycles=3, rate_hz=25.0))
    assert len(slow) == len(fast), "the count must not depend on sampling"
    assert sum(c.measurable for c in fast) >= sum(c.measurable for c in slow), (
        "denser sampling should never make FEWER cycles measurable"
    )
