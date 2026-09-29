"""Tests for rebuilding the annotated video.

Rendering is checked against a synthetic cache rather than a real tracking run,
so these stay fast and need no models. What matters here is the contract: every
source frame is written, overlays come from the stored masks, and the stage
touches nothing but the cache -- that last property is what lets the GPU work
happen once and the annotation be iterated afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from excavator_cycles import masks as mask_io
from excavator_cycles.render import render
from excavator_cycles.track import load_result


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    """A tracking output directory, built by hand: video, masks, track.json."""
    width, height, fps, n_frames = 160, 120, 30.0, 60
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (20 + i, 40), (60 + i, 80), (40, 190, 230), -1)
        writer.write(frame)
    writer.release()

    # Samples at 10 Hz from 30 fps -> every third frame.
    stride, frames, mask_store = 3, [], {}
    for position, frame_index in enumerate(range(0, n_frames, stride)):
        mask = np.zeros((height, width), bool)
        x = 20 + frame_index
        mask[40:80, x : min(x + 40, width)] = True
        mask_store[position] = mask
        frames.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "mask_area_fraction": float(mask.sum() / (width * height)),
                "sam_confidence": 0.9,
                "has_mask": True,
                "detection_box": [float(x), 40.0, float(min(x + 40, width)), 80.0]
                if position % 10 == 0
                else None,
                "detection_score": 0.75 if position % 10 == 0 else None,
                "truck_box": [120.0, 30.0, 150.0, 60.0],
                "detection_truck_iou": 0.02 if position % 10 == 0 else None,
                "detection_mask_iou": 0.9 if position % 10 == 0 else None,
            }
        )

    out = tmp_path / "track_out"
    out.mkdir()
    mask_io.save(out / "masks.npz", mask_store, (height, width))
    (out / "track.json").write_text(
        json.dumps(
            {
                "video": str(video),
                "video_sha256": "0" * 64,
                "config_digest": "deadbeef",
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": n_frames / fps,
                "rate_hz": 10.0,
                "seed": {
                    "sample_position": 0,
                    "frame_index": 0,
                    "time_seconds": 0.0,
                    "box": [20.0, 40.0, 60.0, 80.0],
                    "negative_points": [[130.0, 50.0], [140.0, 55.0]],
                    "truck_box": [120.0, 30.0, 150.0, 60.0],
                },
                "frames": frames,
                "qa": {
                    "status": "pass",
                    "failures": [],
                    "coverage": 1.0,
                    "mask_area": {
                        "median": 0.08,
                        "min": 0.07,
                        "max": 0.09,
                        "samples_far_above_median": 0,
                        "samples_far_below_median": 0,
                    },
                    "detection_agreement_median_iou": 0.9,
                    "sam_confidence_median": 0.9,
                },
            },
            indent=2,
        )
    )
    return out


def test_load_result_roundtrip(fake_cache: Path):
    result, masks = load_result(fake_cache)
    assert result.width == 160
    assert len(result.frames) == len(masks)
    assert result.frames[0].has_mask


def test_render_writes_every_source_frame(fake_cache: Path):
    """Output must be the whole video, not only the sampled frames."""
    stats = render(fake_cache)
    assert stats.output_path.exists()
    assert stats.frames_written == 60, "30 fps x 2 s"
    assert stats.frames_with_mask == 60, "overlays are held between samples"

    written = cv2.VideoCapture(str(stats.output_path))
    ok, frame = written.read()
    written.release()
    assert ok
    assert frame.shape[0] > 120, "a readout panel is appended below the frame"


def test_render_needs_no_model(fake_cache: Path, monkeypatch):
    """The whole point of the cache boundary: no torch, no weights, no GPU."""
    import builtins

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"render must not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    render(fake_cache, out_path=fake_cache / "again.mp4")


def test_render_scale_changes_output_size(fake_cache: Path):
    stats = render(fake_cache, out_path=fake_cache / "big.mp4", scale=2.0)
    cap = cv2.VideoCapture(str(stats.output_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    assert width == 320


def test_render_a_video_larger_than_the_tracked_frames(fake_cache: Path):
    """Tracking downscales to `inference_max_edge`, so the stored masks can be
    smaller than the source video. The render must draw at the tracked size."""
    record = json.loads((fake_cache / "track.json").read_text())
    record["width"], record["height"] = 80, 60
    (fake_cache / "track.json").write_text(json.dumps(record))
    _, masks = load_result(fake_cache)
    half = {i: m[::2, ::2] for i, m in masks.items()}
    mask_io.save(fake_cache / "masks.npz", half, (60, 80))

    stats = render(fake_cache, out_path=fake_cache / "small.mp4", scale=2.0)
    assert stats.frames_written == 60
    assert stats.frames_with_mask == 60, "the masks line up with the frame"
    cap = cv2.VideoCapture(str(stats.output_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    assert width == 160


def test_render_without_boxes(fake_cache: Path):
    stats = render(fake_cache, out_path=fake_cache / "mask_only.mp4", draw_boxes=False)
    assert stats.frames_written == 60


# --- phase banner ---------------------------------------------------------


def _start(phase: str, time: float):
    from excavator_cycles.starts import PhaseStart

    return PhaseStart(phase, time, (time - 0.5, time + 0.5), False, False)


CYCLE_STARTS = [
    _start("digging", 4.0),
    _start("hauling", 10.0),
    _start("dumping", 18.0),
    _start("swinging", 23.0),
    _start("digging", 29.0),
]


def _one_cycle():
    from excavator_cycles.cycles import Cycle

    return Cycle({s.phase: s.time for s in CYCLE_STARTS[:4]}, end=29.0)


def test_phase_at_returns_the_latest_start_that_has_passed():
    from excavator_cycles.render import phase_at

    assert phase_at(CYCLE_STARTS, 0.0) is None, "before the first start, no phase is running"
    assert phase_at(CYCLE_STARTS, 4.0).phase == "digging", "a phase starts AT its start"
    assert phase_at(CYCLE_STARTS, 9.9).phase == "digging"
    assert phase_at(CYCLE_STARTS, 10.0).phase == "hauling"
    assert phase_at(CYCLE_STARTS, 99.0).time == 29.0, "the second dig, not the first"


def test_the_banner_names_the_phase_its_timer_and_the_cycle_count():
    from excavator_cycles.render import banner_text

    assert banner_text(CYCLE_STARTS, [_one_cycle()], 3.0) == [], "nothing before the first dig"
    lines = banner_text(CYCLE_STARTS, [_one_cycle()], 12.0)
    assert lines[0] == "HAULING   2.0s"
    assert lines[1] == "complete cycles: 0"
    assert lines[2] == "this cycle: dig 6.0s  haul 2.0s"


def test_the_cycle_count_goes_up_when_a_cycle_ends():
    """A cycle ends at the next digging start, and not before."""
    from excavator_cycles.render import banner_text

    cycles = [_one_cycle()]
    assert banner_text(CYCLE_STARTS, cycles, 28.9)[1] == "complete cycles: 0"
    assert banner_text(CYCLE_STARTS, cycles, 29.0)[1] == "complete cycles: 1"


def test_the_banner_shows_only_the_current_cycles_phases_so_far():
    from excavator_cycles.render import banner_text

    lines = banner_text(CYCLE_STARTS, [_one_cycle()], 31.0)
    assert lines[0] == "DIGGING   2.0s"
    assert lines[2] == "this cycle: dig 2.0s", "the finished cycle's phases are not repeated"


def test_a_video_that_opens_mid_cycle_shows_no_empty_cycle_line():
    from excavator_cycles.render import banner_text

    lines = banner_text([_start("hauling", 1.0)], [], 2.0)
    assert lines == ["HAULING   1.0s", "complete cycles: 0"]


# --- the physics overlay ------------------------------------------------------
#
# This whole branch was once untested. It is the branch where an option was accepted,
# threaded down two levels and never drawn -- caught by measuring pixels in the
# output video, not by the suite. A test that counts pixels is therefore exactly the
# right shape for it.


@pytest.fixture
def cache_with_features(fake_cache: Path) -> Path:
    """The same cache, plus a stage-3 feature table so `physics=True` has input."""
    from excavator_cycles.features import FeatureTable, Scene, save

    result, _masks = load_result(fake_cache)
    n = len(result.frames)
    times = np.array([f.time_seconds for f in result.frames], dtype=float)
    ramp = np.linspace(-0.3, 0.6, n)
    boxes = np.stack(
        [np.full(n, 20.0), np.full(n, 40.0), np.full(n, 60.0), np.full(n, 80.0)], 1
    )
    table = FeatureTable(
        time_seconds=times,
        bucket_x=np.linspace(0.1, 0.9, n),
        bucket_y=ramp,
        height=ramp,
        dh_dt=np.gradient(ramp, times),
        d2h_dt2=np.zeros(n),
        dx_dt=np.full(n, 0.2),
        speed_x=np.full(n, 0.2),
        rel_cabin_x=np.full(n, 0.3),
        rel_cabin_y=np.full(n, 0.2),
        rel_truck_x=np.full(n, 0.4),
        rel_truck_y=np.full(n, 0.1),
        truck_overlap=np.linspace(0.0, 0.5, n),
        aspect_ratio=np.full(n, 1.2),
        radius=np.full(n, 0.5),
        bucket_box=boxes,
        cabin_box=boxes,
        found=np.ones(n, bool),
    )
    scene = Scene(pivot=(80.0, 60.0), scale=40.0, truck_box=(120.0, 30.0, 150.0, 60.0))
    save(table, scene, fake_cache)
    return fake_cache


def _video_size(path: Path) -> tuple[int, int]:
    """Measured from the written file, not from what the renderer reported."""
    capture = cv2.VideoCapture(str(path))
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def _first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        assert ok, f"could not read {path}"
        return frame
    finally:
        capture.release()


def test_the_physics_overlay_widens_the_canvas(cache_with_features: Path):
    """The graph column only exists when stage 3 has run, so its presence is the
    observable difference -- and it is what `_signal_strip` used to gate, through a
    function that returned its own first argument unchanged."""
    a, b = cache_with_features / "a.mp4", cache_with_features / "b.mp4"
    with_graphs = render(cache_with_features, out_path=a, scale=1.0)
    plain = render(cache_with_features, out_path=b, scale=1.0, physics=False)
    assert _video_size(a)[0] > _video_size(b)[0], (
        f"physics on gave {_video_size(a)}, off gave {_video_size(b)}"
    )
    assert with_graphs.frames_written == plain.frames_written


def test_the_physics_overlay_still_writes_every_frame(cache_with_features: Path):
    """The contract the whole module is built on must survive the extra drawing."""
    stats = render(cache_with_features, out_path=cache_with_features / "p.mp4", scale=1.0)
    assert stats.frames_written == 60


def _frame_at(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok, f"could not read frame {index} of {path}"
        return frame
    finally:
        capture.release()


def _changed(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    return int((a != b).any(axis=2).sum())


def test_the_phase_starts_are_actually_drawn(cache_with_features: Path):
    """The precedent defect, pinned by counting pixels: an option accepted, threaded
    through two functions and never drawn. Nothing in a signature or a docstring can
    catch that; only the output can."""
    bare, drawn = cache_with_features / "bare.mp4", cache_with_features / "drawn.mp4"
    render(cache_with_features, out_path=bare, scale=1.0)
    render(
        cache_with_features,
        out_path=drawn,
        scale=1.0,
        starts=[_start("digging", 0.3), _start("hauling", 1.0)],
    )
    assert _changed(_first_frame(bare), _first_frame(drawn)) > 0


def test_the_search_interval_of_each_start_is_shaded(cache_with_features: Path):
    """The shaded span is how a reader sees where the start was searched for. Moving the
    interval must move the pixels, with the start itself in the same place."""
    from excavator_cycles.starts import PhaseStart

    narrow, wide = cache_with_features / "narrow.mp4", cache_with_features / "wide.mp4"
    render(
        cache_with_features,
        out_path=narrow,
        scale=1.0,
        starts=[PhaseStart("digging", 0.5, (0.45, 0.55), False, False)],
    )
    render(
        cache_with_features,
        out_path=wide,
        scale=1.0,
        starts=[PhaseStart("digging", 0.5, (0.1, 0.9), False, False)],
    )
    assert _changed(_first_frame(narrow), _first_frame(wide)) > 0


def test_the_banner_is_drawn_on_the_picture_once_a_phase_is_running(cache_with_features: Path):
    """At 1 s the digging phase (started at 0.1 s) is running: its banner is drawn in the
    video's top-left corner. The same frame without phases has none."""
    bare, drawn = cache_with_features / "nb.mp4", cache_with_features / "b.mp4"
    render(cache_with_features, out_path=bare, scale=1.0)
    render(cache_with_features, out_path=drawn, scale=1.0, starts=[_start("digging", 0.1)])
    corner = (slice(0, 40), slice(0, 70))
    assert _changed(_frame_at(bare, 30)[corner], _frame_at(drawn, 30)[corner]) > 0
    assert _changed(_frame_at(bare, 0)[corner], _frame_at(drawn, 0)[corner]) == 0, (
        "no banner before the first phase starts"
    )


def test_the_cycle_counter_is_drawn(cache_with_features: Path):
    """A finished cycle changes the banner, so the count must reach the pixels."""
    from excavator_cycles.cycles import Cycle

    starts = [_start("digging", 0.1)]
    none, one = cache_with_features / "c0.mp4", cache_with_features / "c1.mp4"
    render(cache_with_features, out_path=none, scale=1.0, starts=starts, cycles=[])
    finished = Cycle(
        {"digging": 0.0, "hauling": 0.2, "dumping": 0.4, "swinging": 0.6}, end=0.8
    )
    render(cache_with_features, out_path=one, scale=1.0, starts=starts, cycles=[finished])
    corner = (slice(0, 40), slice(0, 110))
    assert _changed(_frame_at(none, 40)[corner], _frame_at(one, 40)[corner]) > 0
