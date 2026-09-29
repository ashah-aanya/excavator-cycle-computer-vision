"""The command line: the `cycles` stage, and the `run` chain a reviewer types.

Everything reachable without a GPU pass is covered here. Video rendering needs the source
clip and the mask cache, neither of which belongs in a fixture, so the `render` stage is
checked only for what it is handed (`tests/test_render.py` covers the drawing itself).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from excavator_cycles import cli
from excavator_cycles.cycles import PHASES, read_phases

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dev_clip"

# What the phase search gives on the provided video: five phase starts, one cycle.
EXPECTED = {
    "cycle_count": 1,
    "average_cycle_duration_seconds": 24.721,
    "average_phase_duration_seconds": {
        "digging": 5.705,
        "hauling": 10.109,
        "dumping": 2.202,
        "swinging": 6.706,
    },
}


@pytest.fixture
def features_dir(tmp_path):
    """A directory holding stage 3's output for the provided video."""
    directory = tmp_path / "clip"
    directory.mkdir()
    for name in ("features.npz", "scene.json"):
        shutil.copyfile(FIXTURE / name, directory / name)
    return directory


def _cycles_args(track_dir, out=None):
    return argparse.Namespace(track_dir=track_dir, out=out)


# --- the cycles stage ------------------------------------------------------------


def test_cycles_writes_the_answer_the_method_gives_on_a_real_feature_table(features_dir):
    assert cli._cmd_cycles(_cycles_args(features_dir)) == 0
    answer = json.loads((features_dir / "answer.json").read_text())
    assert answer["cycle_count"] == EXPECTED["cycle_count"]
    assert answer["average_cycle_duration_seconds"] == pytest.approx(
        EXPECTED["average_cycle_duration_seconds"], abs=0.0015
    )
    for phase, seconds in EXPECTED["average_phase_duration_seconds"].items():
        assert answer["average_phase_duration_seconds"][phase] == pytest.approx(
            seconds, abs=0.0015
        ), phase


def test_cycles_writes_the_answer_where_it_is_told(features_dir, tmp_path):
    out = tmp_path / "elsewhere.json"
    cli._cmd_cycles(_cycles_args(features_dir, out))
    assert json.loads(out.read_text())["cycle_count"] == 1


def test_cycles_also_writes_the_phases_the_video_is_drawn_from(features_dir):
    cli._cmd_cycles(_cycles_args(features_dir))
    starts, cycles = read_phases(features_dir / "phases.json")
    assert [s.phase for s in starts] == [
        "digging",
        "hauling",
        "dumping",
        "swinging",
        "digging",
    ]
    assert len(cycles) == 1


def test_cycles_says_what_it_found(features_dir, capsys):
    cli._cmd_cycles(_cycles_args(features_dir))
    out = capsys.readouterr().out
    assert "complete cycles    : 1" in out
    for phase in PHASES:
        assert phase in out


def _without_a_truck(features_dir):
    arrays = dict(np.load(features_dir / "features.npz"))
    arrays["rel_truck_x"] = np.full_like(arrays["rel_truck_x"], np.nan)
    np.savez_compressed(features_dir / "features.npz", **arrays)


def test_a_video_with_no_truck_still_gets_an_answer_that_says_no_cycles(features_dir, caplog):
    """The method needs a truck to measure against. A video without one is a finding,
    not a crash: the file must exist, with every field, and the reason must be kept."""
    _without_a_truck(features_dir)
    assert cli._cmd_cycles(_cycles_args(features_dir)) == 0
    answer = json.loads((features_dir / "answer.json").read_text())
    assert answer["cycle_count"] == 0
    assert answer["average_cycle_duration_seconds"] == 0.0
    assert set(answer["average_phase_duration_seconds"]) == set(PHASES)
    assert "no truck" in caplog.text
    reason = json.loads((features_dir / "phases.json").read_text())["stop"]["why"]
    assert "no truck" in reason


def test_the_command_line_has_no_label_arguments():
    """Labels only ever score a result, from outside the pipeline. The commands that run
    it must not even offer to read them."""
    for argv in (["cycles", "dir", "--labels", "x"], ["run", "v.mp4", "--labels", "x"]):
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == 2, argv


# --- the render stage ------------------------------------------------------------


@pytest.fixture
def drawn(monkeypatch):
    """Replace the drawing with a recorder, and return what it was handed."""
    import excavator_cycles.render as render_module

    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(output_path="x.mp4", frames_written=1, frames_with_mask=1)

    monkeypatch.setattr(render_module, "render", fake)
    return seen


def _render_args(track_dir):
    return argparse.Namespace(track_dir=track_dir, out=None, scale=1.0, no_boxes=False)


def test_render_is_handed_the_phases_and_cycles_that_cycles_found(features_dir, drawn):
    cli._cmd_cycles(_cycles_args(features_dir))
    cli._cmd_render(_render_args(features_dir))
    assert len(drawn["starts"]) == 5
    assert len(drawn["cycles"]) == 1


def test_render_still_works_when_no_phases_were_found_yet(features_dir, drawn):
    """`render` on a directory straight from `track`: masks and boxes, no phases."""
    cli._cmd_render(_render_args(features_dir))
    assert drawn["starts"] is None and drawn["cycles"] is None


# --- the `run` chain -------------------------------------------------------------
#
# `run` is the one command the help text calls "THE DELIVERABLE": video in, answer.json
# and an annotated video out. It calls the other stages through their own `_cmd_*`
# functions, so what a reviewer runs is what these tests exercise.


def _run_args(video, out, **kw):
    base = dict(
        video=video,
        config=None,
        out=out,
        device=None,
        detector="grounding_dino",
        rate=None,
        reuse=True,
        scale=2.0,
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def out_dir(tmp_path):
    """A results directory already holding stage 3's output, so `--reuse` skips the GPU."""
    directory = tmp_path / "results"
    directory.mkdir()
    for name in ("features.npz", "scene.json"):
        shutil.copyfile(FIXTURE / name, directory / name)
    # `masks.npz` need only EXIST for `--reuse` to skip tracking.
    (directory / "masks.npz").write_bytes(b"")
    return directory


@pytest.fixture
def stub_render(monkeypatch):
    """The render stage needs real masks and the source clip; record its arguments."""
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_cmd_render", lambda args: calls.append(args) or 0)
    return calls


def test_run_writes_the_answer_and_asks_for_the_video_beside_it(out_dir, stub_render):
    assert cli._cmd_run(_run_args(Path("clip.mp4"), out_dir)) == 0
    assert json.loads((out_dir / "answer.json").read_text())["cycle_count"] == 1
    assert (out_dir / "phases.json").exists()
    (call,) = stub_render
    assert call.out == out_dir / "annotated.mp4"
    assert call.track_dir == out_dir


def test_run_replaces_a_stale_answer(out_dir, stub_render):
    (out_dir / "answer.json").write_text(json.dumps({"cycle_count": 999, "STALE": True}))
    cli._cmd_run(_run_args(Path("clip.mp4"), out_dir))
    written = json.loads((out_dir / "answer.json").read_text())
    assert "STALE" not in written and written["cycle_count"] == 1


def test_run_puts_its_results_in_outputs_named_for_the_video_by_default(
    tmp_path, monkeypatch, stub_render
):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "outputs" / "clip"
    results.mkdir(parents=True)
    for name in ("features.npz", "scene.json"):
        shutil.copyfile(FIXTURE / name, results / name)
    (results / "masks.npz").write_bytes(b"")
    assert cli._cmd_run(_run_args(Path("some/where/clip.mp4"), None)) == 0
    assert (results / "answer.json").exists()


def test_run_continues_past_a_track_qa_concern_and_still_answers(
    out_dir, monkeypatch, stub_render, capsys
):
    """`track` returns non-zero for a QA VERDICT, having written the masks anyway.

    Aborting turned one tracker wobble on a hidden video into zero for every graded
    field, which is strictly worse than a flagged answer. The concern must stay
    visible in the output, and the exit code stays 0 because the answer and video
    exist: a non-zero status reads as "no result" to a caller and could discard them.
    """
    # Remove the tracking product so the (stubbed) track stage actually runs. A stage
    # that runs makes everything after it run too, so features is stubbed to leave the
    # fixture's `features.npz` in place rather than needing real masks.
    (out_dir / "masks.npz").unlink()
    monkeypatch.setattr(cli, "_cmd_track", lambda args: 1)
    monkeypatch.setattr(cli, "_cmd_features", lambda args: 0)
    status = cli._cmd_run(_run_args(Path("clip.mp4"), out_dir))
    assert json.loads((out_dir / "answer.json").read_text())["cycle_count"] == 1
    assert len(stub_render) == 1, "a QA concern must not cost the video either"
    assert status == 0, "the answer exists, so the run has not failed"
    assert "CONCERN: track QA flagged this run" in capsys.readouterr().out


def test_run_stops_when_a_later_stage_genuinely_fails(out_dir, monkeypatch, stub_render):
    """Only `track`'s status is advisory. A features or cycles failure is fatal."""
    (out_dir / "features.npz").unlink()
    monkeypatch.setattr(cli, "_cmd_features", lambda args: 3)
    assert cli._cmd_run(_run_args(Path("clip.mp4"), out_dir)) == 3
    assert not (out_dir / "answer.json").exists()
    assert stub_render == [], "nothing after the failed stage may run"


def test_run_skips_a_stage_whose_product_is_already_there(out_dir, monkeypatch, stub_render):
    """`--reuse` is what makes iterating on the later stages bearable; if it silently
    re-ran the GPU pass it would be worse than useless."""
    called: list[str] = []
    monkeypatch.setattr(cli, "_cmd_track", lambda args: called.append("track") or 0)
    monkeypatch.setattr(cli, "_cmd_features", lambda args: called.append("features") or 0)
    cli._cmd_run(_run_args(Path("clip.mp4"), out_dir))
    assert called == [], f"both products exist, so neither stage should run; ran {called}"


def test_reuse_reruns_every_stage_after_one_that_ran(out_dir, monkeypatch, stub_render):
    """`--reuse` asked each stage alone whether its product existed, so fresh masks could
    be followed by a REUSED `features.npz` built from the old ones -- an answer from
    neither run. Once a stage runs, everything downstream is stale."""
    (out_dir / "masks.npz").unlink()
    called: list[str] = []
    monkeypatch.setattr(cli, "_cmd_track", lambda args: called.append("track") or 0)
    monkeypatch.setattr(cli, "_cmd_features", lambda args: called.append("features") or 0)
    cli._cmd_run(_run_args(Path("clip.mp4"), out_dir))
    assert called == ["track", "features"], called
