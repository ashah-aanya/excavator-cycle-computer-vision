"""Tests for ``eval/score.py``.

A scorer that is wrong is worse than no scorer: it either hides a real failure or
manufactures a fake one. So these tests pin the boundary behaviour -- an answer
just inside the tolerance must pass, one just outside must fail -- and the exit
codes, because that is what a human or a CI job actually reads.

``eval/`` is deliberately not importable as a package (the pipeline must not be
able to reach the labels), so the module is loaded by path here. The CLI-level
tests shell out, which is the only honest way to check an exit code.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCORE_PATH = REPO / "eval" / "score.py"
LABELS_PATH = REPO / "eval" / "labels.json"

CORRECT_ANSWER = {
    "cycle_count": 1,
    "average_cycle_duration_seconds": 25.188,
    "average_phase_duration_seconds": {
        "digging": 6.572,
        "hauling": 7.907,
        "dumping": 4.404,
        "swinging": 6.305,
    },
}


def _load_score_module():
    """Import eval/score.py by path -- eval/ is not on the import path by design."""
    spec = importlib.util.spec_from_file_location("eval_score", SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_mod = _load_score_module()


def _write(tmp_path: Path, payload, name: str = "answer.json") -> Path:
    path = tmp_path / name
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return path


def _run(answer: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCORE_PATH), str(answer), "--labels", str(LABELS_PATH), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


# ------------------------------------------------------------------ reading labels


def test_expected_comes_from_the_labels_file():
    expected = score_mod.load_expected(LABELS_PATH)
    assert expected["cycle_count"] == 1
    assert expected["average_cycle_duration_seconds"] == pytest.approx(25.188)
    assert expected["average_phase_duration_seconds.digging"] == pytest.approx(6.572)
    assert expected["average_phase_duration_seconds.swinging"] == pytest.approx(6.305)
    assert len(expected) == 6


def test_tolerance_comes_from_the_labels_not_from_the_code():
    assert score_mod.load_tolerance(LABELS_PATH) == 0.6


def test_score_module_does_not_import_the_pipeline():
    """Standard library only: the scorer must run without the pipeline installed."""
    text = SCORE_PATH.read_text(encoding="utf-8")
    assert "excavator_cycles" not in text.replace("src/excavator_cycles/", "")


# ------------------------------------------------------------------------ verdicts


def test_exactly_correct_answer_passes(tmp_path: Path):
    result = _run(_write(tmp_path, CORRECT_ANSWER))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL: PASS" in result.stdout
    assert "FAIL" not in result.stdout


def test_answer_inside_tolerance_passes(tmp_path: Path):
    """0.5 s off on one field is within the graded +/-0.6 s, so it must pass."""
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    answer["average_phase_duration_seconds"]["hauling"] += 0.5
    result = _run(_write(tmp_path, answer))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL: PASS" in result.stdout
    assert "+0.500" in result.stdout  # the error is reported signed


def test_answer_outside_tolerance_fails_on_that_field_only(tmp_path: Path):
    """0.9 s off on one field: that field fails, the others must not."""
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    answer["average_phase_duration_seconds"]["dumping"] -= 0.9
    result = _run(_write(tmp_path, answer))
    assert result.returncode == 1
    assert "OVERALL: FAIL" in result.stdout
    failing = [ln for ln in result.stdout.splitlines() if ln.endswith("FAIL")]
    assert len(failing) == 1
    assert failing[0].startswith("average_phase_duration_seconds.dumping")
    assert "-0.900" in failing[0]


@pytest.mark.parametrize("delta,expected_code", [(0.599, 0), (0.601, 1)])
def test_tolerance_boundary(tmp_path: Path, delta: float, expected_code: int):
    """The +/-0.6 s edge itself, from both sides."""
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    answer["average_cycle_duration_seconds"] += delta
    result = _run(_write(tmp_path, answer, f"answer_{delta}.json"))
    assert result.returncode == expected_code, result.stdout + result.stderr


def test_wrong_cycle_count_fails(tmp_path: Path):
    """A count has no tolerance: 2 cycles is not 1 cycle, however close the times."""
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    answer["cycle_count"] = 2
    result = _run(_write(tmp_path, answer))
    assert result.returncode == 1
    assert "OVERALL: FAIL" in result.stdout
    cycle_line = next(ln for ln in result.stdout.splitlines() if ln.startswith("cycle_count"))
    assert cycle_line.endswith("FAIL")


def test_cycle_count_is_not_scored_with_the_seconds_tolerance(tmp_path: Path):
    """Guards the obvious bug: 1 vs 2 differs by 1, which is > 0.6, but 1 vs 1.5
    would pass a numeric tolerance. It must not."""
    expected = score_mod.load_expected(LABELS_PATH)
    actual = dict(expected)
    actual["cycle_count"] = 1.5
    results = score_mod.score(expected, actual, 0.6)
    assert not next(r for r in results if r.field == "cycle_count").passed


def test_every_field_is_scored(tmp_path: Path):
    """All five graded fields appear in the table, plus cycle_count."""
    result = _run(_write(tmp_path, CORRECT_ANSWER))
    for field in (
        "cycle_count",
        "average_cycle_duration_seconds",
        "average_phase_duration_seconds.digging",
        "average_phase_duration_seconds.hauling",
        "average_phase_duration_seconds.dumping",
        "average_phase_duration_seconds.swinging",
    ):
        assert field in result.stdout


# -------------------------------------------------------------------- bad input


def test_missing_answer_file(tmp_path: Path):
    result = _run(tmp_path / "does_not_exist.json")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "not found" in result.stderr


def test_malformed_json(tmp_path: Path):
    result = _run(_write(tmp_path, "{not json,,,", "broken.json"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "not valid JSON" in result.stderr


def test_answer_is_a_list_not_an_object(tmp_path: Path):
    result = _run(_write(tmp_path, [1, 2, 3], "list.json"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "must be a JSON object" in result.stderr


def test_missing_required_field(tmp_path: Path):
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    del answer["average_phase_duration_seconds"]["swinging"]
    result = _run(_write(tmp_path, answer, "incomplete.json"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "swinging" in result.stderr


def test_non_numeric_value(tmp_path: Path):
    answer = json.loads(json.dumps(CORRECT_ANSWER))
    answer["average_cycle_duration_seconds"] = "25.188"
    result = _run(_write(tmp_path, answer, "stringy.json"))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "must be a number" in result.stderr


def test_missing_labels_file(tmp_path: Path):
    answer = _write(tmp_path, CORRECT_ANSWER)
    result = subprocess.run(
        [
            sys.executable,
            str(SCORE_PATH),
            str(answer),
            "--labels",
            str(tmp_path / "nope.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "labels file not found" in result.stderr
