"""Cycles and the answer: the arithmetic the task is graded on.

A cycle runs from one digging start to the next. Each phase lasts until the next one
starts, and swinging lasts until the next dig. The averages are over complete cycles
only, and the file has exactly the fields and names the task specifies.
"""

from __future__ import annotations

import json

import pytest

from excavator_cycles.cycles import (
    PHASES,
    Answer,
    Cycle,
    assemble,
    read_phases,
    summarise,
    write_answer,
    write_phases,
)
from excavator_cycles.starts import PhaseSearch, PhaseStart


def start(phase: str, time: float, low_agreement: bool = False) -> PhaseStart:
    return PhaseStart(phase, time, (time - 0.5, time + 0.5), False, low_agreement)


def one_cycle(dig: float = 2.0, haul: float = 5.0, dump: float = 9.0, swing: float = 12.0):
    return [
        start("digging", dig),
        start("hauling", haul),
        start("dumping", dump),
        start("swinging", swing),
    ]


# --- durations -----------------------------------------------------------------


def test_each_phase_lasts_until_the_next_one_starts():
    cycle = Cycle({"digging": 2.0, "hauling": 5.0, "dumping": 9.0, "swinging": 12.0}, end=17.0)
    assert cycle.durations() == {
        "digging": 3.0,
        "hauling": 4.0,
        "dumping": 3.0,
        "swinging": 5.0,
    }


def test_swinging_runs_to_the_next_digging_start_not_to_where_it_stops_moving():
    """The spec: swinging *ends immediately before the next digging phase begins*."""
    cycle = Cycle({"digging": 0.0, "hauling": 1.0, "dumping": 2.0, "swinging": 3.0}, end=10.0)
    assert cycle.durations()["swinging"] == 7.0


def test_the_four_phases_add_up_to_the_cycle():
    cycle = Cycle(
        {"digging": 2.0, "hauling": 5.5, "dumping": 9.25, "swinging": 12.0}, end=19.0
    )
    assert sum(cycle.durations().values()) == pytest.approx(cycle.duration)
    assert cycle.duration == 17.0


# --- which spans count as cycles -------------------------------------------------


def test_a_dig_to_the_next_dig_with_all_four_phases_is_a_cycle():
    cycles = assemble([*one_cycle(), start("digging", 17.0)])
    assert len(cycles) == 1
    assert cycles[0].end == 17.0
    assert cycles[0].starts["hauling"] == 5.0


def test_one_digging_start_bounds_no_cycle():
    """A cycle needs the dig that closes it. A lone dig is the head or tail of one."""
    assert assemble(one_cycle()) == []


def test_footage_before_the_first_dig_and_after_the_last_is_ignored():
    """The spec: ignore any incomplete cycle at the beginning or end of the video."""
    starts = [
        start("swinging", 0.5),  # the tail of a cycle that began before the video
        *one_cycle(dig=2.0, haul=5.0, dump=9.0, swing=12.0),
        start("digging", 17.0),
        start("hauling", 20.0),  # the head of a cycle that the video ends inside
    ]
    cycles = assemble(starts)
    assert len(cycles) == 1
    assert cycles[0].starts["digging"] == 2.0 and cycles[0].end == 17.0


def test_a_span_missing_a_phase_is_dropped_and_the_next_cycle_still_counts():
    starts = [
        start("digging", 2.0),
        start("hauling", 5.0),  # no dump, no swing: the search abandoned this cycle
        *one_cycle(dig=17.0, haul=20.0, dump=24.0, swing=27.0),
        start("digging", 32.0),
    ]
    cycles = assemble(starts)
    assert len(cycles) == 1
    assert cycles[0].starts["digging"] == 17.0 and cycles[0].end == 32.0


def test_two_complete_cycles_in_a_row_share_the_middle_dig():
    starts = [*one_cycle(), *one_cycle(dig=17.0, haul=20.0, dump=24.0, swing=27.0)]
    starts.append(start("digging", 32.0))
    cycles = assemble(starts)
    assert [c.end for c in cycles] == [17.0, 32.0]


# --- the answer ------------------------------------------------------------------


def test_the_answer_averages_every_phase_and_the_cycle_over_the_complete_cycles():
    first = Cycle({"digging": 0.0, "hauling": 2.0, "dumping": 6.0, "swinging": 8.0}, end=12.0)
    second = Cycle(
        {"digging": 12.0, "hauling": 16.0, "dumping": 18.0, "swinging": 22.0}, end=30.0
    )
    answer = summarise([first, second])
    assert answer.cycle_count == 2
    assert answer.average_phase_duration_seconds == {
        "digging": 3.0,  # (2 + 4) / 2
        "hauling": 3.0,  # (4 + 2) / 2
        "dumping": 3.0,  # (2 + 4) / 2
        "swinging": 6.0,  # (4 + 8) / 2
    }
    assert answer.average_cycle_duration_seconds == 15.0  # (12 + 18) / 2


def test_no_complete_cycle_is_reported_as_zeros_not_an_error():
    """A clip shorter than one cycle has nothing to average; the file must still exist."""
    answer = summarise([])
    assert answer.cycle_count == 0
    assert answer.average_cycle_duration_seconds == 0.0
    assert answer.average_phase_duration_seconds == dict.fromkeys(PHASES, 0.0)


def test_the_answer_has_exactly_the_fields_the_task_specifies():
    """The task's schema, key for key, and nothing else."""
    answer = Answer(
        1, 15.2, {"digging": 2.501, "hauling": 3.342, "dumping": 4.234, "swinging": 5.123}
    )
    record = answer.to_dict()
    assert list(record) == [
        "cycle_count",
        "average_cycle_duration_seconds",
        "average_phase_duration_seconds",
    ]
    assert list(record["average_phase_duration_seconds"]) == [
        "digging",
        "hauling",
        "dumping",
        "swinging",
    ]


def test_durations_are_written_to_three_decimals_and_the_count_is_an_integer():
    answer = Answer(2, 24.72149999, dict.fromkeys(PHASES, 5.7054999))
    record = answer.to_dict()
    assert record["average_cycle_duration_seconds"] == 24.721
    assert record["average_phase_duration_seconds"]["digging"] == 5.705
    assert isinstance(record["cycle_count"], int)


def test_write_answer_writes_readable_json(tmp_path):
    answer = summarise(assemble([*one_cycle(), start("digging", 17.0)]))
    path = write_answer(answer, tmp_path / "answer.json")
    text = path.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == answer.to_dict()
    assert json.loads(text)["cycle_count"] == 1


# --- what the annotated video is drawn from --------------------------------------


def test_the_phases_survive_a_round_trip_through_phases_json(tmp_path):
    starts = [
        *one_cycle(),
        start("digging", 17.0, low_agreement=True),
    ]
    cycles = assemble(starts)
    search = PhaseSearch(starts=starts, gaps=[], stop=None)
    path = write_phases(search, cycles, tmp_path / "phases.json")

    read_starts, read_cycles = read_phases(path)
    assert read_starts == starts
    assert read_cycles == cycles
