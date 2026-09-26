"""The command line, which had no tests at all.

Why this file exists
--------------------
Changing `locate` to return `Onset` objects instead of `(phase, time)` tuples broke
`--test` outright -- `for phase, when in onsets` raised `TypeError: cannot unpack
non-iterable Onset object` -- and the full suite stayed green, because nothing
exercised the diagnostic path. The pipeline's own output was fine; the thing a
reviewer runs to JUDGE that output was dead on arrival.

That is a worse failure than a wrong number, because a reviewer who cannot run the
diagnostic has no way to see that a number is wrong. The two tests below cover the
shape of the bug: the printing path over a mix of refined and unrefined onsets, and
the command wiring from a real feature table through to `answer.json`.

Video rendering is deliberately NOT covered here -- it needs the source clip and the
mask cache, neither of which belongs in a fixture. `--test`'s render call is still
exercised by hand; these cover everything reachable without a GPU pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from excavator_cycles.cli import _cmd_cycles, _print_breakdown
from excavator_cycles.cycles import Cycle
from excavator_cycles.fsm import Onset

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dev_clip"
PHASES = ("digging", "hauling", "dumping", "swinging")


def test_the_breakdown_prints_an_onset_whose_refinement_failed(capsys):
    """The exact break: `Onset` is not a 2-tuple.

    And an unrefined onset must be VISIBLE rather than skipped -- it is the case a
    reader most needs to see, since the cycle is still counted and the missing
    onset is why the averages are empty.
    """
    onsets = [
        Onset("digging", None, 3.40),
        Onset("hauling", 11.91, 12.71),
        Onset("dumping", 13.71, 13.51),
        Onset("swinging", 25.52, 25.72),
    ]
    _print_breakdown([], onsets, truth={})
    out = capsys.readouterr().out
    assert "digging" in out
    assert "pass 2 found nothing" in out, "a failed refinement must be reported, not dropped"
    assert "3.40" in out, "the coarse time is what is left to show"
    assert "11.91" in out


def test_the_breakdown_prints_errors_against_the_truth(capsys):
    """The `--labels` column. Truth is optional, so it has its own path."""
    onsets = [Onset("hauling", 11.91, 12.71)]
    _print_breakdown([], onsets, truth={"hauling": 10.74})
    out = capsys.readouterr().out
    assert "10.74" in out
    assert "+1.17" in out, "the signed error is the number being judged"


def test_the_breakdown_explains_an_excluded_cycle(capsys):
    """`reason` is a sentence for a reader, so it has to actually reach them."""
    cycle = Cycle(
        onsets={"hauling": 1.0, "dumping": 2.0, "swinging": 3.0},  # digging missing
        ends=4.0,
        span=(0.0, 4.0),
        occurred=set(PHASES),
    )
    _print_breakdown([cycle], [], truth={})
    out = capsys.readouterr().out
    assert "EXCLUDED" in out
    assert "digging occurred but its onset was not located" in out


def test_the_breakdown_says_so_when_there_are_no_cycles(capsys):
    """Silence would read as "no output yet" rather than "nothing to report"."""
    _print_breakdown([], [], truth={})
    assert "fewer than two digging onsets" in capsys.readouterr().out


def test_cycles_writes_an_answer_from_a_real_feature_table(tmp_path):
    """The command's whole job, wired end to end without a GPU.

    Asserts the schema the task specifies, exactly, and that the count is the one
    the clip contains -- this is the regression that `cycle_count: 0` was.
    """
    out = tmp_path / "answer.json"
    args = argparse.Namespace(
        track_dir=FIXTURE,
        config=None,
        out=out,
        test=False,
        labels=None,
        scale=2.0,
    )
    assert _cmd_cycles(args) == 0

    answer = json.loads(out.read_text())
    assert set(answer) == {
        "cycle_count",
        "average_cycle_duration_seconds",
        "average_phase_duration_seconds",
    }
    assert set(answer["average_phase_duration_seconds"]) == set(PHASES)
    assert answer["cycle_count"] == 1, "the dev clip holds exactly one complete cycle"
    assert isinstance(answer["average_cycle_duration_seconds"], float)


def test_cycles_prints_the_levels_it_derived(tmp_path, capsys):
    """A reviewer has to be able to see what "low" and "moving" were taken to mean,
    including which side the truck was decided to be on."""
    args = argparse.Namespace(
        track_dir=FIXTURE,
        config=None,
        out=tmp_path / "answer.json",
        test=False,
        labels=None,
        scale=2.0,
    )
    _cmd_cycles(args)
    out = capsys.readouterr().out
    for expected in ("low height", "over truck", "moving", "dump side", "cycles occurred"):
        assert expected in out, f"the run summary must mention {expected!r}"


def test_the_unrefined_onsets_are_reported_as_warnings(caplog):
    """A silently missing onset is how `cycle_count: 0` went unnoticed for a branch.

    The dev clip's two digging refinements fail, and that must be loud -- it is the
    reason four of the five graded fields are zero.
    """
    import logging

    args = argparse.Namespace(
        track_dir=FIXTURE,
        config=None,
        out=Path("/dev/null"),
        test=False,
        labels=None,
        scale=2.0,
    )
    with caplog.at_level(logging.WARNING):
        _cmd_cycles(args)
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("cue found no" in m for m in warnings), (
        f"expected a refinement warning: {warnings}"
    )
    assert any("none could be measured" in m for m in warnings)


@pytest.mark.parametrize("phase", PHASES)
def test_every_phase_appears_in_the_answer_even_when_it_could_not_be_measured(tmp_path, phase):
    """The schema is fixed by the task: all four keys, always, zero if unmeasured."""
    out = tmp_path / "answer.json"
    _cmd_cycles(
        argparse.Namespace(
            track_dir=FIXTURE, config=None, out=out, test=False, labels=None, scale=2.0
        )
    )
    assert phase in json.loads(out.read_text())["average_phase_duration_seconds"]
