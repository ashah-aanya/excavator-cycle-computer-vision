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
