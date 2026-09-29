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


@pytest.mark.parametrize("fps,rate", [(30.0, 10.0), (25.0, 12.5), (24.0, 12.0), (10.0, 10.0)])
def test_actual_rate_is_what_the_stride_delivers(tmp_path: Path, fps: float, rate: float):
    """A whole-frame stride cannot hit every request, so the rate that is RECORDED
    must be the one delivered: 10 Hz asked of 25 fps is 12.5 Hz, and the real gaps show it."""
    clip = _write_video(tmp_path / f"rate{fps}.mp4", fps=fps, n_frames=int(fps * 3))
    info = probe(clip)
    assert info.actual_rate(10.0) == pytest.approx(rate)
    times = [s.time_seconds for s in iter_samples(clip, rate_hz=10.0)]
    assert np.diff(times) == pytest.approx(1 / rate, rel=1e-3)


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


# --- every time comes from the file ------------------------------------------


def _write_uneven_video(path: Path, times: list[float], size=(64, 48)) -> Path:
    """A clip whose frames are stored at exactly ``times`` (seconds), like a screen
    recording that captures faster in some stretches than others."""
    from fractions import Fraction

    import av

    base = Fraction(1, 1000)
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("mpeg4", rate=30)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = base
        for i, t in enumerate(times):
            image = np.full((size[1], size[0], 3), (i * 20) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = round(t * 1000)
            frame.time_base = base
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


# 20 frames: 10 at 0.02 s spacing (fast), then 10 at 0.1 s spacing (slow).
UNEVEN = [round(0.02 * i, 3) for i in range(10)] + [
    round(0.18 + 0.1 * i, 3) for i in range(1, 11)
]


def test_uneven_frames_get_their_stored_times(tmp_path):
    """The bug this fixes. Frame 10 is stored at 0.28 s; `index / average fps`
    (20 frames over 1.28 s) puts it at 0.64 s, 0.36 s late on a 1.3 s clip. The
    stored timestamp is exact."""
    from excavator_cycles.video import frame_times

    path = _write_uneven_video(tmp_path / "uneven.mp4", UNEVEN)
    got = frame_times(path)
    assert got == pytest.approx(UNEVEN, abs=1e-3)
    samples = list(iter_samples(path, rate_hz=1000.0))  # every frame
    assert [s.time_seconds for s in samples] == pytest.approx(UNEVEN, abs=1e-3)
    average = len(UNEVEN) / (UNEVEN[-1] + 0.1)
    assert 10 / average - UNEVEN[10] == pytest.approx(0.36, abs=0.01), "index / fps is off"


def test_an_uneven_file_is_reported_variable(tmp_path):
    info = probe(_write_uneven_video(tmp_path / "uneven.mp4", UNEVEN), verify=True)
    assert info.timeline == "variable"
    assert "stored in the file" in info.summary()


def test_an_even_file_is_reported_constant_and_timed_by_its_timestamps(tmp_path):
    from excavator_cycles.video import frame_times

    path = _write_video(tmp_path / "cfr.mp4", fps=25.0, n_frames=40)
    info = probe(path, verify=True)
    assert info.timeline == "constant"
    assert frame_times(path) == pytest.approx(np.arange(40) / 25.0, abs=1e-6)


def test_the_task_videos_stored_timestamps_are_thirtieths():
    """Pinned against the real file when it is present.

    The header's rate is 29.97396912 (from the same header that claims 1102
    frames when 886 exist), but every stored timestamp is exactly n/30. An earlier
    fix (PR #2) trusted the header and computed index / 29.974; the file itself
    says 30, and the file is what we read now. The two differ by at most 0.026 s
    over the clip."""
    from excavator_cycles.video import frame_times

    video = (
        Path(__file__).resolve().parent.parent / "construction_excavator_cycle_duration_1.mp4"
    )
    if not video.exists():
        # SKIPS IN CI: `*.mp4` is gitignored. The behaviour (times come from the
        # file) is covered by the synthetic tests above.
        pytest.skip("task video not present")
    times = frame_times(video)
    assert len(times) == 886
    assert times == pytest.approx(np.arange(886) / 30.0, abs=1e-9)
