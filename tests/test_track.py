"""Tests for mask storage, prompt construction, seeding and quality judgement.

No models run here. What is tested is the logic that decides *how* the models are
used -- which frame to prompt, where to put the negative points, whether the
resulting masks look trustworthy -- because that logic is ours, and it is where
a mistake would be silent.

The scenarios mirror real footage: roughly half the frames have the excavator
box merged with the truck box, which is what was measured on the task video.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles import masks as mask_io
from excavator_cycles.config import Config
from excavator_cycles.detect.base import Detection
from excavator_cycles.track import (
    DetectionPass,
    FrameRecord,
    choose_seed,
    estimate_truck_box,
    evaluate_quality,
    negative_points,
)


def detection(box, score=0.8) -> Detection:
    return Detection(
        boxes=np.array([box], dtype=np.float32),
        scores=np.array([score], dtype=np.float32),
        labels=["excavator."],
    )


# --- mask storage -----------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((8, 8), bool),
        np.ones((8, 8), bool),
        np.eye(16, dtype=bool),
        np.random.default_rng(0).random((32, 48)) > 0.7,
    ],
)
def test_rle_roundtrip(mask):
    restored = mask_io.decode(mask_io.encode(mask), mask.shape)
    assert np.array_equal(restored, mask)


def test_rle_handles_mask_starting_true():
    """The encoding assumes runs start with False; a True-first mask must still work."""
    mask = np.array([[True, True, False, False]], dtype=bool)
    assert np.array_equal(mask_io.decode(mask_io.encode(mask), mask.shape), mask)


def test_rle_is_much_smaller_than_raw(tmp_path):
    """The whole point: a machine-shaped island in a field of False compresses hard."""
    rng = np.random.default_rng(0)
    frames = {}
    for i in range(50):
        m = np.zeros((272, 480), bool)
        x = 100 + int(rng.integers(-20, 20))
        m[120:200, x : x + 120] = True
        frames[i] = m

    path = tmp_path / "masks.npz"
    mask_io.save(path, frames, (272, 480))
    raw_bytes = 50 * 272 * 480
    assert path.stat().st_size < raw_bytes / 20

    loaded, shape = mask_io.load(path)
    assert shape == (272, 480)
    assert np.array_equal(loaded[7], frames[7])


# --- truck box and negative points ------------------------------------------


def make_pass(clean_boxes, merged_boxes):
    """Frames where the boxes are separate, plus frames where they are identical."""
    detections = DetectionPass()
    position = 0
    for exc, truck in clean_boxes:
        detections.excavator[position] = detection(exc)
        detections.truck[position] = detection(truck)
        position += 1
    for box in merged_boxes:
        detections.excavator[position] = detection(box, score=0.95)  # merged often scores high
        detections.truck[position] = detection(box, score=0.5)
        position += 1
    return detections


def test_truck_box_ignores_merged_frames():
    """A merged frame's "truck" box is the merged box and must not pollute the estimate."""
    config = Config.load()
    clean = [([10, 10, 60, 60], [200, 100, 300, 160])] * 3
    merged = [[10, 10, 300, 160]] * 5  # would drag the estimate far left if used

    box = estimate_truck_box(make_pass(clean, merged), config)
    assert box is not None
    assert box.tolist() == [200, 100, 300, 160]


def test_truck_box_none_when_never_separated():
    """No truck ever cleanly seen is legitimate -- prompt without negatives."""
    config = Config.load()
    assert estimate_truck_box(make_pass([], [[10, 10, 300, 160]] * 4), config) is None


def test_negative_points_sit_low_in_the_truck_box():
    """Placement is not incidental: a negative on the bucket cancels the exclusion.

    The bucket empties into the bed, so it occupies the upper part of the truck
    box. Every point must be in the lower portion.
    """
    config = Config.load()
    truck = np.array([100.0, 200.0, 200.0, 300.0])  # 100x100 box
    points = negative_points(truck, config)

    assert len(points) == config.track.negative_rows * config.track.negative_columns
    for x, y in points:
        assert 100 < x < 200, "inside the box horizontally"
        assert y >= 200 + 100 * 0.5, "in the lower half, away from where the bucket goes"
        assert y <= 300


def test_no_negative_points_without_a_truck():
    assert negative_points(None, Config.load()) == []


# --- seeding ----------------------------------------------------------------


def test_seed_prefers_an_unmerged_frame_even_at_lower_score():
    """A merged frame often scores higher; confidence alone would pick the wrong one."""
    config = Config.load()
    detections = DetectionPass()
    detections.excavator[0] = detection([10, 10, 60, 60], score=0.60)  # clean, modest
    detections.truck[0] = detection([200, 100, 300, 160])
    detections.excavator[1] = detection([10, 10, 300, 160], score=0.95)  # merged, confident
    detections.truck[1] = detection([10, 10, 300, 160])

    position, box = choose_seed(detections, config)
    assert position == 0
    assert box.tolist() == [10, 10, 60, 60]


def test_seed_falls_back_to_a_merged_frame():
    """If every frame is merged, seed anyway -- negative points make it usable."""
    config = Config.load()
    detections = make_pass([], [[10, 10, 300, 160]] * 3)
    position, _ = choose_seed(detections, config)
    assert position in detections.excavator


def test_seed_raises_when_nothing_was_ever_detected():
    """A stage-1 failure must be loud, not silently produce garbage masks."""
    with pytest.raises(RuntimeError, match="never found the excavator"):
        choose_seed(DetectionPass(), Config.load())


# --- quality judgement ------------------------------------------------------


def records(areas, confidence=0.9, agreement=0.85):
    out = []
    for i, area in enumerate(areas):
        out.append(
            FrameRecord(
                frame_index=i * 3,
                time_seconds=i * 0.1,
                mask_area_fraction=area,
                sam_confidence=confidence,
                has_mask=area > 0,
                detection_mask_iou=agreement if i % 10 == 0 else None,
                detection_truck_iou=0.02 if i % 10 == 0 else None,
            )
        )
    return out


def test_quality_passes_on_steady_masks():
    qa = evaluate_quality(records([0.07] * 100), Config.load())
    assert qa["status"] == "pass"
    assert qa["coverage"] == 1.0
    assert qa["mask_area"]["median"] == pytest.approx(0.07)


def test_quality_flags_masks_that_double():
    """Roughly what absorbing the truck looks like: 7% -> 14%."""
    qa = evaluate_quality(records([0.07] * 70 + [0.14] * 30), Config.load())
    assert qa["status"] == "fail"
    assert any("above the median" in f for f in qa["failures"])


def test_quality_flags_masks_that_collapse():
    qa = evaluate_quality(records([0.07] * 70 + [0.01] * 30), Config.load())
    assert qa["status"] == "fail"
    assert any("below the median" in f for f in qa["failures"])


def test_quality_flags_missing_masks():
    qa = evaluate_quality(records([0.07] * 80 + [0.0] * 20), Config.load())
    assert qa["status"] == "fail"
    assert any("have a mask" in f for f in qa["failures"])


def test_quality_flags_disagreement_with_detections():
    """The independent opinion: if the mask is elsewhere, something is wrong."""
    qa = evaluate_quality(records([0.07] * 100, agreement=0.2), Config.load())
    assert qa["status"] == "fail"
    assert any("disagrees" in f for f in qa["failures"])


def test_quality_tolerates_a_few_odd_frames():
    """Occupancy dips when the bucket is buried; that is physics, not failure."""
    qa = evaluate_quality(records([0.07] * 95 + [0.02] * 5), Config.load())
    assert qa["status"] == "pass"
