"""Tests for the hand-made ground truth in ``eval/labels.json``.

The labels are typed by hand, which means their arithmetic can be typed wrong by
hand. These tests recompute every derived number from the two things that are
genuinely primary -- the boundary frames and the frame rate -- so a fat-fingered
duration cannot sit in the file looking authoritative.

They also assert the *separation*: the pipeline must not be able to reach these
labels, so nothing under ``src/excavator_cycles/`` may mention them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO / "eval" / "labels.json"

# The real rate of the task video. Not 30 -- see the module docstring in
# src/excavator_cycles/video.py for why the 0.1% matters at a +/-0.6 s tolerance.
FPS = 29.97396912419384

EXPECTED_SECONDS = {
    "digging": 6.572,
    "hauling": 7.907,
    "dumping": 4.404,
    "swinging": 6.305,
    "cycle": 25.188,
}
EXPECTED_FRAMES = {
    "digging": 197,
    "hauling": 237,
    "dumping": 132,
    "swinging": 189,
    "cycle": 755,
}


@pytest.fixture(scope="module")
def labels() -> dict:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def test_video_facts(labels: dict):
    """fps and frame count are the two inputs every duration depends on."""
    video = labels["video"]
    assert video["filename"] == "construction_excavator_cycle_duration_1.mp4"
    assert video["fps"] == FPS
    # 886 frames decode; the container header claims 1102 and is wrong.
    assert video["frame_count"] == 886
    assert video["frame_index_range"] == [0, 885]


def test_boundary_frames(labels: dict):
    assert labels["boundaries"]["digging_begins"] == 125
    assert labels["boundaries"]["hauling_begins"] == 322
    assert labels["boundaries"]["dumping_begins"] == 559
    assert labels["boundaries"]["swinging_begins"] == 691
    assert labels["boundaries"]["cycle_ends"] == 880


def test_boundaries_are_strictly_increasing(labels: dict):
    """Phases tile the cycle in order, so the onsets must too."""
    order = [
        "digging_begins",
        "hauling_begins",
        "dumping_begins",
        "swinging_begins",
        "cycle_ends",
    ]
    frames = [labels["boundaries"][k] for k in order]
    assert frames == sorted(frames)
    assert len(set(frames)) == len(frames)


@pytest.mark.parametrize("phase", ["digging", "hauling", "dumping", "swinging", "cycle"])
def test_duration_in_frames_is_the_difference_of_its_boundaries(labels: dict, phase: str):
    entry = labels["durations"][phase]
    assert entry["to_frame"] - entry["from_frame"] == entry["frames"]
    assert entry["frames"] == EXPECTED_FRAMES[phase]


@pytest.mark.parametrize("phase", ["digging", "hauling", "dumping", "swinging", "cycle"])
def test_seconds_follow_from_frames_and_fps(labels: dict, phase: str):
    entry = labels["durations"][phase]
    assert entry["seconds_exact"] == pytest.approx(entry["frames"] / FPS, abs=1e-9)
    assert entry["seconds"] == EXPECTED_SECONDS[phase]
    # The graded value is the exact one at 3 dp -- within a millisecond either way,
    # which allows the deliberate 25.1885 -> 25.188 truncation documented in the file
    # (rounding would give 25.189 and break the sum-to-cycle property below).
    assert entry["seconds"] == pytest.approx(entry["seconds_exact"], abs=1e-3)


def test_phases_tile_the_cycle(labels: dict):
    """No gaps and no overlaps: the four phases must sum to the cycle exactly.

    This is the property the design leans on (docs/pipeline-design.md 2.3) and the
    reason the cycle is reported as 25.188 rather than the rounded 25.189.
    """
    durations = labels["durations"]
    phases = ["digging", "hauling", "dumping", "swinging"]
    assert sum(durations[p]["frames"] for p in phases) == durations["cycle"]["frames"]
    assert sum(durations[p]["seconds"] for p in phases) == pytest.approx(
        durations["cycle"]["seconds"], abs=1e-9
    )


def test_cycle_fits_inside_the_decodable_frames(labels: dict):
    """The one complete cycle has to be inside the 886 frames that actually decode."""
    assert labels["cycle_count"] == 1
    assert labels["durations"]["cycle"]["from_frame"] >= 0
    assert labels["durations"]["cycle"]["to_frame"] <= labels["video"]["frame_count"]


def test_expected_answer_matches_the_durations(labels: dict):
    """The answer-shaped block must not drift from the durations it summarises."""
    expected = labels["expected_answer"]
    assert expected["cycle_count"] == labels["cycle_count"] == 1
    assert (
        expected["average_cycle_duration_seconds"] == labels["durations"]["cycle"]["seconds"]
    )
    for phase in ("digging", "hauling", "dumping", "swinging"):
        assert (
            expected["average_phase_duration_seconds"][phase]
            == labels["durations"][phase]["seconds"]
        )


def test_tolerance_is_the_graded_one(labels: dict):
    assert labels["scoring"]["tolerance_seconds"] == 0.6


def test_provenance_says_hand_made_not_model_made(labels: dict):
    """These are Aanya's labels. Anything that reads them must know that."""
    source = labels["provenance"]["source"].lower()
    assert "hand" in source
    assert "aanya" in source


def test_asymmetric_offset_caveat_is_recorded(labels: dict):
    """The +9 / +22 frame asymmetry is the one thing a reader must not miss.

    It puts ~0.43 s of slack into swinging and into the cycle average, which is
    most of the +/-0.6 s budget on two of the five graded fields.
    """
    caveats = " ".join(labels["known_caveats"]).lower()
    assert "asymmetric" in caveats
    assert "0.43" in caveats
    assert "swinging" in caveats
    conventions = " ".join(labels["conventions"]).lower()
    assert "+9 frames" in conventions


def test_pipeline_never_references_the_labels():
    """Structural separation, enforced rather than promised.

    The task spec forbids the pipeline reading the answer, so no module under
    src/excavator_cycles/ may name these files at all.
    """
    offenders = []
    for path in (REPO / "src" / "excavator_cycles").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(
            name in text
            for name in (
                "eval/labels",
                "labels.json",
                "labels_long_clip",
                "eval.score",
                "check_cues",
            )
        ):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"pipeline modules reference the eval labels: {offenders}"
