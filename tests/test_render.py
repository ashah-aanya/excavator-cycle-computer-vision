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


def test_render_without_boxes(fake_cache: Path):
    stats = render(fake_cache, out_path=fake_cache / "mask_only.mp4", draw_boxes=False)
    assert stats.frames_written == 60


# --- phase banner ---------------------------------------------------------


def test_phase_at_returns_the_latest_onset_that_has_passed():
    from excavator_cycles.render import phase_at

    onsets = {"digging": 4.0, "hauling": 10.0, "dumping": 18.0, "swinging": 23.0}
    assert phase_at(onsets, 0.0) is None, "before the first onset, no phase is running"
    assert phase_at(onsets, 4.0) == "digging", "a phase starts AT its onset"
    assert phase_at(onsets, 9.9) == "digging"
    assert phase_at(onsets, 10.0) == "hauling"
    assert phase_at(onsets, 99.0) == "swinging"


def test_phase_at_ignores_onsets_that_were_not_found():
    from excavator_cycles.render import phase_at

    assert phase_at({"digging": 4.0, "hauling": None}, 20.0) == "digging"
    assert phase_at({"digging": None}, 20.0) is None


# --- the physics overlay ------------------------------------------------------
#
# This whole branch was untested. It is the branch where `render(reference=...)`
# was accepted, threaded down two levels and never drawn -- caught by measuring
# pixels in the output video, not by the suite. A test that counts pixels is
# therefore exactly the right shape for it.


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


def test_the_reference_onsets_are_actually_drawn(cache_with_features: Path):
    """The precedent defect, pinned by counting pixels.

    `reference=` was accepted, threaded through two functions and never drawn. The
    ground truth was simply absent from the video while the caller believed it was
    there -- and the author described the prediction lines to the user as her own
    labels. Nothing in a signature or a docstring can catch that; only the output
    can.
    """
    without = cache_with_features / "no_ref.mp4"
    with_ref = cache_with_features / "ref.mp4"
    render(cache_with_features, out_path=without, scale=1.0)
    render(
        cache_with_features,
        out_path=with_ref,
        scale=1.0,
        reference={"digging": 0.3, "hauling": 0.9, "dumping": 1.2, "swinging": 1.5},
    )

    plain, marked = _first_frame(without), _first_frame(with_ref)
    assert plain.shape == marked.shape
    changed = int((plain != marked).any(axis=2).sum())
    assert changed > 0, "passing `reference=` changed nothing in the output at all"


def test_the_predicted_onsets_are_drawn_too(cache_with_features: Path):
    """Same check for `onsets=`, so the pair cannot silently diverge."""
    bare = cache_with_features / "bare.mp4"
    drawn = cache_with_features / "drawn.mp4"
    render(cache_with_features, out_path=bare, scale=1.0)
    render(
        cache_with_features, out_path=drawn, scale=1.0, onsets={"digging": 0.3, "hauling": 1.0}
    )

    assert int((_first_frame(bare) != _first_frame(drawn)).any(axis=2).sum()) > 0


def test_the_windows_are_shaded(cache_with_features: Path):
    """The shaded span is how a reader sees where pass 1 searched."""
    bare = cache_with_features / "nw.mp4"
    shaded = cache_with_features / "w.mp4"
    render(cache_with_features, out_path=bare, scale=1.0)
    render(
        cache_with_features,
        out_path=shaded,
        scale=1.0,
        windows=[("digging", 0.2, 0.5), ("hauling", 0.8, 1.1)],
    )

    assert int((_first_frame(bare) != _first_frame(shaded)).any(axis=2).sum()) > 0


def test_the_calibrated_levels_are_drawn(cache_with_features: Path):
    """`levels=` is the same shape of option as `reference=`, which was once
    accepted, threaded and never drawn. So it gets the same pixel-count check."""
    bare = cache_with_features / "nl.mp4"
    drawn = cache_with_features / "l.mp4"
    render(cache_with_features, out_path=bare, scale=1.0)
    render(
        cache_with_features,
        out_path=drawn,
        scale=1.0,
        levels={"height": ("Otsu", 0.1), "truck_overlap": ("Otsu", 0.25)},
    )

    assert int((_first_frame(bare) != _first_frame(drawn)).any(axis=2).sum()) > 0
