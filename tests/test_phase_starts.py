"""The phase search on the two fixture clips.

The method was written as scripts under `eval/` and then moved into `src/`. The move must
not change a single number, so `tests/fixtures/expected_phase_starts.json` holds what the
ORIGINAL scripts returned, and these tests check `src/` still returns it. Nothing here
reads a label: the expected values are the method's own earlier output.

The other tests state properties any correct search has: every start lies inside the
interval it was picked from, the starts run forward in time, footage filmed from the
other side gives the same answer, and a video with no truck is refused clearly instead
of crashing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from excavator_cycles.cues import NoPhases
from excavator_cycles.features import load
from excavator_cycles.starts import find_phase_starts

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXPECTED = json.loads((FIXTURES / "expected_phase_starts.json").read_text())
CLIPS = ("dev_clip", "long_clip")
ROUNDING = 0.002  # the expected values were saved to three decimals


@pytest.fixture(scope="module")
def searches():
    return {clip: find_phase_starts(*load(FIXTURES / clip)) for clip in CLIPS}


@pytest.mark.parametrize("clip", CLIPS)
def test_the_starts_are_what_the_original_scripts_found(searches, clip):
    got = [(s.phase, s.time) for s in searches[clip].starts]
    want = EXPECTED[clip]["starts"]
    assert [p for p, _ in got] == [p for p, _ in want], "the phases found, in order"
    for (phase, time), (_, expected) in zip(got, want, strict=True):
        assert time == pytest.approx(expected, abs=ROUNDING), phase


@pytest.mark.parametrize("clip", CLIPS)
def test_the_windows_are_what_the_original_scripts_found(searches, clip):
    got = [(s.phase, s.window) for s in searches[clip].starts]
    want = EXPECTED[clip]["windows"]
    assert len(got) == len(want)
    for (phase, window), (expected_phase, expected) in zip(got, want, strict=True):
        assert phase == expected_phase
        assert window == pytest.approx(tuple(expected), abs=ROUNDING), phase


@pytest.mark.parametrize("clip", CLIPS)
def test_where_the_search_stopped_and_what_it_abandoned(searches, clip):
    search = searches[clip]
    stopped = search.stop["phase"] if search.stop else None
    assert stopped == EXPECTED[clip]["stopped_at"]
    assert len(search.gaps) == EXPECTED[clip]["abandoned_cycles"]


@pytest.mark.parametrize("clip", CLIPS)
def test_every_start_is_inside_the_interval_it_was_picked_from(searches, clip):
    for s in searches[clip].starts:
        assert s.window[0] - 1e-9 <= s.time <= s.window[1] + 1e-9, s


@pytest.mark.parametrize("clip", CLIPS)
def test_the_starts_run_forward_and_phases_follow_the_cycle(searches, clip):
    starts = searches[clip].starts
    times = [s.time for s in starts]
    assert times == sorted(times) and len(set(times)) == len(times)
    order = ("digging", "hauling", "dumping", "swinging")
    assert [s.phase for s in starts] == [order[i % 4] for i in range(len(starts))]


def test_the_83_second_clip_holds_three_cycles_and_the_last_dig_closes_the_third(searches):
    phases = [s.phase for s in searches["long_clip"].starts]
    assert phases.count("digging") == 4, "three cycles need four digging starts"


def test_mirrored_footage_reads_the_same():
    """Filmed from the other side, every horizontal sign flips. The truck side and the
    pile-to-truck line are read from the video, so the starts must not move."""
    table, scene = load(FIXTURES / "dev_clip")
    width = 500.0
    box = np.asarray(table.bucket_box, dtype=float)
    mirrored = dataclasses.replace(
        table,
        dx_dt=-table.dx_dt,
        bucket_x=-table.bucket_x,
        rel_cabin_x=-table.rel_cabin_x,
        rel_truck_x=-table.rel_truck_x,
        bucket_box=np.stack(
            [width - box[:, 2], box[:, 1], width - box[:, 0], box[:, 3]], axis=1
        ),
    )
    x1, y1, x2, y2 = scene.truck_box
    flipped_scene = dataclasses.replace(scene, truck_box=(width - x2, y1, width - x1, y2))

    original = find_phase_starts(table, scene).starts
    flipped = find_phase_starts(mirrored, flipped_scene).starts
    assert [s.phase for s in original] == [s.phase for s in flipped]
    assert [s.time for s in flipped] == pytest.approx([s.time for s in original], abs=1e-6)


def test_a_video_with_no_truck_is_refused_with_a_reason():
    """Every rule measures the bucket against the dump truck. Without one there is
    nothing to search, and the caller must be told why rather than crashed."""
    table, scene = load(FIXTURES / "dev_clip")
    no_truck = dataclasses.replace(table, rel_truck_x=np.full(len(table), np.nan))
    with pytest.raises(NoPhases, match="no truck"):
        find_phase_starts(no_truck, scene)
