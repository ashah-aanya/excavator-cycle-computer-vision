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

from excavator_cycles.video import iter_samples, probe


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


# --- which clock do we trust? ---------------------------------------------


def test_a_constant_rate_file_is_classified_constant(tmp_path):
    """And therefore timed by index/fps, not by the decoder.

    CAP_PROP_POS_MSEC is often computed from a ROUNDED rate rather than read
    from the container: on the task video it returns exactly n/30 while the true
    rate is 29.97396912, so every timestamp is 0.1% short. That is 0.022 s over
    one work cycle -- small against a 0.6 s tolerance, but systematic, and it
    offsets every prediction against ground truth converted at the true rate.
    """
    from excavator_cycles.video import classify_timeline

    path = _write_video(tmp_path / "cfr.mp4", fps=25.0, n_frames=40)
    assert classify_timeline(path) == "constant"


def test_constant_rate_timestamps_come_from_the_frame_rate(tmp_path):
    from excavator_cycles.video import probe

    info = probe(_write_video(tmp_path / "cfr.mp4", fps=25.0, n_frames=40))
    assert info.timeline == "constant"
    assert "index / fps" in info.summary()


def test_constant_rate_timing_is_exact_arithmetic_not_the_decoder(tmp_path):
    """On "constant" the helper must compute index/fps and never consult the
    decoder -- that is the whole point, since the decoder may be quantised."""
    import cv2

    from excavator_cycles.video import _timestamp_seconds

    path = _write_video(tmp_path / "cfr.mp4", fps=30.0, n_frames=10)
    capture = cv2.VideoCapture(str(path))
    try:
        for index in (0, 3, 7):
            got = _timestamp_seconds(capture, index, 29.97396912419384, "constant")
            assert got == pytest.approx(index / 29.97396912419384), (
                "constant-rate timing must be exact index/fps"
            )
    finally:
        capture.release()


def test_the_two_clocks_disagree_on_the_task_video():
    """The bug this fixes, pinned against the real file when it is present.

    The container reports 29.97396912 fps; OpenCV's POS_MSEC returns exactly
    n/30. Over one 755-frame work cycle that is 0.022 s -- 3.6% of the budget,
    in the same direction every time.
    """
    import cv2

    video = (
        Path(__file__).resolve().parent.parent / "construction_excavator_cycle_duration_1.mp4"
    )
    if not video.exists():
        pytest.skip("task video not present")
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        stamps = []
        for _ in range(60):
            if not capture.grab():
                break
            stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        capture.release()
    implied = 1000 * (len(stamps) - 1) / (stamps[-1] - stamps[0])
    assert implied == pytest.approx(30.0, abs=0.01), "POS_MSEC is on a round 30 fps timeline"
    assert fps == pytest.approx(29.97396912, abs=1e-6), "the container knows better"
    assert abs(755 / 30 - 755 / fps) == pytest.approx(0.022, abs=0.002)
