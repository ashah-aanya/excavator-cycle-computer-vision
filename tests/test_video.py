"""Tests for video reading.

The point of these is the frames-to-seconds boundary. Every hidden-video failure
mode we worried about in the design starts with a rule that secretly assumed
30 fps, so the conversion gets tested directly, on synthetic videos at several
frame rates.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from excavator_cycles.video import iter_samples, probe, sample_times


def _write_video(path: Path, fps: float, n_frames: int, size=(160, 120)) -> Path:
    """A tiny synthetic clip: each frame is a solid shade of its own index."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def clip_30fps(tmp_path: Path) -> Path:
    return _write_video(tmp_path / "clip30.mp4", fps=30.0, n_frames=90)  # 3 s


def test_probe_reads_metadata(clip_30fps: Path):
    info = probe(clip_30fps)
    assert info.width == 160 and info.height == 120
    assert info.fps == pytest.approx(30.0, abs=0.01)
    assert info.frame_count == 90
    assert info.duration_seconds == pytest.approx(3.0, abs=0.05)


def test_probe_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


@pytest.mark.parametrize("fps,expected_stride", [(30.0, 3), (25.0, 2), (10.0, 1)])
def test_stride_adapts_to_frame_rate(tmp_path: Path, fps: float, expected_stride: int):
    """The SAME 10 Hz request must mean different frame counts at different fps.

    This is the whole reason rates are expressed in Hz rather than in frames.
    """
    clip = _write_video(tmp_path / f"clip{fps}.mp4", fps=fps, n_frames=int(fps * 2))
    info = probe(clip)
    assert info.stride_for(10.0) == expected_stride


def test_stride_never_zero(tmp_path: Path):
    """A video slower than the requested rate is sampled every frame, not skipped."""
    clip = _write_video(tmp_path / "slow.mp4", fps=5.0, n_frames=10)
    assert probe(clip).stride_for(10.0) == 1


def test_samples_are_spaced_in_time(clip_30fps: Path):
    samples = list(iter_samples(clip_30fps, rate_hz=10.0))
    assert len(samples) == pytest.approx(30, abs=2)  # ~3 s at 10 Hz

    times = [s.time_seconds for s in samples]
    assert times == sorted(times)
    gaps = np.diff(times)
    assert np.allclose(gaps, 0.1, atol=0.02)


def test_trimming(clip_30fps: Path):
    samples = list(iter_samples(clip_30fps, rate_hz=10.0, start_seconds=1.0, end_seconds=2.0))
    assert samples, "expected samples inside the trim window"
    assert all(1.0 <= s.time_seconds <= 2.0 for s in samples)


def test_downscaling_preserves_aspect_ratio(clip_30fps: Path):
    samples = list(iter_samples(clip_30fps, rate_hz=2.0, max_edge=80))
    image = samples[0].image
    assert max(image.shape[:2]) == 80
    assert image.shape[1] / image.shape[0] == pytest.approx(160 / 120, abs=0.05)


def test_sample_times_spread_across_video(clip_30fps: Path):
    info = probe(clip_30fps)
    times = sample_times(info, count=5)
    assert len(times) == 5
    assert times == sorted(times)
    # Spread across the clip, but clear of the very first and last frames.
    assert 0 < times[0] < 0.2
    assert info.duration_seconds - 0.2 < times[-1] < info.duration_seconds


def test_probe_verify_counts_decodable_frames(clip_30fps: Path):
    """With verify, the frame count comes from demuxing rather than the header.

    The task video's header claims 1102 frames when 886 decode -- a 20% overstated
    duration. Phase durations are safe because they use real per-frame timestamps,
    but anything derived from duration would inherit the error silently.
    """
    header = probe(clip_30fps)
    verified = probe(clip_30fps, verify=True)

    assert verified.frame_count == 90
    assert verified.duration_seconds == pytest.approx(3.0, abs=0.05)
    # For a well-formed file the two agree; the point is that verify does not
    # depend on the header being honest.
    assert verified.frame_count == header.frame_count


def test_probe_verify_matches_what_iteration_yields(clip_30fps: Path):
    """The verified count must equal the frames actually reachable."""
    info = probe(clip_30fps, verify=True)
    assert len(list(iter_samples(clip_30fps, rate_hz=30.0))) == pytest.approx(
        info.frame_count, abs=1
    )
