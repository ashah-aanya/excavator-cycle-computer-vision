"""Tests for provenance capture.

The thing under test is a promise: after any stage runs, the directory beside
its outputs explains what produced them. That promise has to hold when the stage
fails, too -- a failed run's record is the more useful one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from excavator_cycles.config import Config
from excavator_cycles.provenance import (
    Environment,
    GitState,
    config_digest,
    file_digest,
    run_record,
)


def test_git_state_captured():
    state = GitState.capture()
    # Tests may run from a tarball with no git, so absence is allowed --
    # but if a revision is reported it must look like a SHA.
    if state.revision is not None:
        assert len(state.revision) == 40
        assert state.dirty in (True, False)


def test_environment_records_versions():
    env = Environment.capture()
    assert env.python.startswith("3.")
    assert "numpy" in env.packages  # a core dependency is always present
    assert env.device  # never empty; names cpu/mps/cuda


def test_file_digest_depends_on_content(tmp_path: Path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"excavator")
    b.write_bytes(b"excavator")  # same content, different name
    c.write_bytes(b"bulldozer")

    assert file_digest(a) == file_digest(b)
    assert file_digest(a) != file_digest(c)


def test_config_digest_ignores_key_order():
    assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})


def test_config_digest_changes_with_any_threshold():
    """The cache key must notice a changed threshold, or stale results get reused."""
    base = Config.load().to_dict()
    tweaked = Config.load(overrides={"features": {"min_sample_confidence": 0.9}}).to_dict()
    assert config_digest(base) != config_digest(tweaked)


def test_run_record_written_on_success(tmp_path: Path):
    config = Config.load().to_dict()
    with run_record("demo", tmp_path, config) as record:
        record.metrics["detection_rate"] = 0.98
        record.outputs.append("frames/000.jpg")

    data = json.loads((tmp_path / "run.json").read_text())
    assert data["stage"] == "demo"
    assert data["status"] == "ok"
    assert data["metrics"]["detection_rate"] == 0.98
    assert data["duration_seconds"] >= 0
    assert data["config"]["sampling"]["rate_hz"] == 10.0
    assert data["config_digest"]


def test_run_record_written_on_failure(tmp_path: Path):
    """A crashed stage must still leave an explanation behind."""
    with (
        pytest.raises(RuntimeError, match="detector exploded"),
        run_record("demo", tmp_path, Config.load().to_dict()),
    ):
        raise RuntimeError("detector exploded")

    data = json.loads((tmp_path / "run.json").read_text())
    assert data["status"] == "failed"
    assert "detector exploded" in data["error"]


def test_run_record_hashes_inputs(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")

    out = tmp_path / "out"
    with run_record("demo", out, Config.load().to_dict()) as record:
        record.add_input("video", video)

    data = json.loads((out / "run.json").read_text())
    assert data["inputs"]["video"]["sha256"] == file_digest(video)
    assert data["inputs"]["video"]["bytes"] == len(b"not really a video")
