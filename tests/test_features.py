"""Kinematic features (design diagram stage 2).

Most of these pin **sign conventions**. Image y increases downward, so a flipped
sign in `height` would invert the digging-to-hauling trigger and the pipeline
would still run, still produce plausible numbers, and be wrong. That failure is
silent, so it gets the most tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.features import (
    FeatureTable,
    Scene,
    build_features,
    load,
    overlap_fraction,
    save,
    truck_box,
)
from excavator_cycles.track import FrameRecord, TrackResult

CONFIG = Config.load()
SHAPE = (120, 200)


def _record(index: int, *, truck=None, iou=None) -> FrameRecord:
    return FrameRecord(
        frame_index=index * 3,
        time_seconds=index * 0.1,
        has_mask=True,
        truck_box=truck,
        detection_truck_iou=iou,
    )


def _result(n: int, **kw) -> TrackResult:
    return TrackResult(
        video="x.mp4",
        video_sha256="0" * 64,
        config_digest="d",
        width=SHAPE[1],
        height=SHAPE[0],
        fps=30.0,
        duration_seconds=n * 0.1,
        rate_hz=10.0,
        seed={},
        frames=[_record(i, **kw) for i in range(n)],
        qa={},
    )


def _blob(cx, cy, half=6):
    m = np.zeros(SHAPE, dtype=bool)
    m[cy - half : cy + half + 1, cx - half : cx + half + 1] = True
    return m


def _scene(n=12):
    """A machine whose body never moves and whose bucket climbs steadily."""
    excavator, bucket = {}, {}
    for i in range(n):
        body = _blob(50, 90, 14)
        b = _blob(120, 90 - 3 * i, 5)  # rising: y DECREASES
        excavator[i] = body | b
        bucket[i] = b
    return excavator, bucket


# --- sign conventions -----------------------------------------------------


def test_height_is_up_positive():
    """A bucket whose image y decreases is RISING, so height must increase."""
    excavator, bucket = _scene()
    table, _ = build_features(_result(12), {"excavator": excavator, "bucket": bucket}, CONFIG)
    assert table.height[-1] > table.height[0], "bucket rose; height must rise"
    assert np.nanmean(table.dh_dt[3:]) > 0, "dh/dt must be positive while rising"


def test_rel_cabin_y_is_positive_when_the_bucket_is_above_the_cabin():
    excavator, bucket = _scene()
    table, _ = build_features(_result(12), {"excavator": excavator, "bucket": bucket}, CONFIG)
    assert table.rel_cabin_y[-1] > 0, "bucket ends above the cabin"


def test_dx_dt_is_right_positive():
    excavator, bucket = {}, {}
    for i in range(12):
        b = _blob(60 + 4 * i, 60, 5)  # moving right
        excavator[i] = _blob(50, 90, 14) | b
        bucket[i] = b
    table, _ = build_features(_result(12), {"excavator": excavator, "bucket": bucket}, CONFIG)
    assert np.nanmean(table.dx_dt[3:]) > 0
    assert np.all(table.speed_x[np.isfinite(table.speed_x)] >= 0)


def test_there_is_no_dy_dt_column():
    """Deliberate: dh_dt and a dy_dt would differ only in sign, and having both
    is how a trigger ends up reading the wrong one."""
    assert "dy_dt" not in FeatureTable.__dataclass_fields__
    assert "dh_dt" in FeatureTable.__dataclass_fields__


# --- scale invariance -----------------------------------------------------


def test_height_is_dimensionless_because_it_is_divided_by_reach():
    """Both terms are pixel lengths over L, so the result cannot be in pixels."""
    excavator, bucket = _scene()
    table, scene = build_features(
        _result(12), {"excavator": excavator, "bucket": bucket}, CONFIG
    )
    assert scene.scale > 1.0, "L is measured in pixels and should be sizeable"
    assert np.nanmax(np.abs(table.height)) < 10.0, "height is in units of L, not pixels"
    assert np.nanmax(np.abs(table.bucket_x)) < 10.0


# --- the truck ------------------------------------------------------------


def test_truck_box_is_the_median_of_the_cleanly_separated_detections():
    frames = [
        _record(0, truck=[10.0, 10.0, 20.0, 20.0], iou=0.0),
        _record(1, truck=[12.0, 12.0, 22.0, 22.0], iou=0.0),
        _record(2, truck=[14.0, 14.0, 24.0, 24.0], iou=0.0),
    ]
    result = _result(3)
    result.frames = frames
    assert truck_box(result, CONFIG) == (12.0, 12.0, 22.0, 22.0)


def test_merged_detections_are_ignored():
    """A frame where the detector merged the two machines says nothing."""
    frames = [
        _record(0, truck=[10.0, 10.0, 20.0, 20.0], iou=0.0),
        _record(1, truck=[900.0, 900.0, 999.0, 999.0], iou=0.95),  # merged; must not count
    ]
    result = _result(2)
    result.frames = frames
    assert truck_box(result, CONFIG) == (10.0, 10.0, 20.0, 20.0)


def test_no_truck_is_a_legitimate_outcome():
    result = _result(3)
    assert truck_box(result, CONFIG) is None


def test_no_truck_makes_the_truck_features_nan_not_zero():
    """Zero would read as 'the bucket is exactly over the truck'."""
    excavator, bucket = _scene()
    table, scene = build_features(
        _result(12), {"excavator": excavator, "bucket": bucket}, CONFIG
    )
    assert scene.truck_box is None
    assert np.isnan(table.truck_overlap).all()
    assert np.isnan(table.rel_truck_x).all()


# --- overlap --------------------------------------------------------------


def test_overlap_is_the_share_of_the_bucket_not_of_the_union():
    """A small bucket wholly inside a big truck box is 1.0, not a small ratio."""
    assert overlap_fraction((10, 10, 20, 20), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_overlap_of_disjoint_boxes_is_zero():
    assert overlap_fraction((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_overlap_counts_pixels_inclusively():
    """Boxes are inclusive, so x0=0..x1=10 is ELEVEN columns, not ten.

    The covered part here is columns 5..10 -- six of the eleven -- over all
    eleven rows: 6*11 / 11*11 = 0.5454..., not 0.5. Getting this wrong
    under-reports every partial overlap by a few percent.
    """
    assert overlap_fraction((0, 0, 10, 10), (5, 0, 50, 10)) == pytest.approx(6 * 11 / 121)


def test_a_one_pixel_tall_bucket_on_the_bed_is_fully_overlapped():
    """The regression that matters: the bucket is thinnest in projection at
    exactly the moment it tips over to dump, which is the event overlap gates.
    Exclusive arithmetic computed 0 * n / 0 here and reported NO overlap."""
    assert overlap_fraction((10, 10, 20, 10), (0, 0, 100, 100)) == pytest.approx(1.0)
    assert overlap_fraction((10, 10, 10, 20), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_a_single_pixel_bucket_is_not_a_division_by_zero():
    assert overlap_fraction((7, 7, 7, 7), (0, 0, 100, 100)) == pytest.approx(1.0)
    assert overlap_fraction((7, 7, 7, 7), (50, 50, 60, 60)) == 0.0


def test_boxes_touching_at_one_column_do_overlap():
    """They share a real pixel column, so the answer is not zero."""
    assert overlap_fraction((0, 0, 10, 10), (10, 0, 20, 10)) == pytest.approx(1 * 11 / 121)


# --- structure ------------------------------------------------------------


def test_too_few_samples_raises_rather_than_returning_nonsense():
    with pytest.raises(ValueError, match="at least 2 samples"):
        build_features(_result(1), {"excavator": {0: _blob(50, 50)}}, CONFIG)


def test_missing_excavator_masks_raise():
    with pytest.raises(ValueError, match="no excavator masks"):
        build_features(_result(4), {}, CONFIG)


def test_round_trip_through_disk(tmp_path):
    excavator, bucket = _scene()
    table, scene = build_features(
        _result(12), {"excavator": excavator, "bucket": bucket}, CONFIG
    )
    save(table, scene, tmp_path)
    again, scene_again = load(tmp_path)
    assert np.allclose(again.height, table.height, equal_nan=True)
    assert scene_again.pivot == pytest.approx(scene.pivot)
    assert scene_again.scale == pytest.approx(scene.scale)


def test_scene_round_trips_without_a_truck():
    scene = Scene(pivot=(1.0, 2.0), scale=10.0, truck_box=None)
    assert Scene.from_dict(scene.to_dict()).truck_box is None


# --- the confidence floor -----------------------------------------------------


def _confident_result(n: int, confidences: list[float]) -> TrackResult:
    """A run where each sample's bucket mask carries a stated confidence."""
    import dataclasses

    base = _result(n)
    frames = [
        dataclasses.replace(record, has_bucket_mask=True, bucket_confidence=confidences[i])
        for i, record in enumerate(base.frames)
    ]
    return dataclasses.replace(base, frames=frames)


def test_a_low_confidence_bucket_mask_is_treated_as_missing():
    """`min_sample_confidence` was documented from the start and never read.

    A low-confidence mask is a GAP IN PERCEPTION, not a measurement. Keeping one
    lets a bad mask set the box, and every rate derived from that box then carries
    an excursion the machine never made -- which is worse than a hole, because a
    hole is visible in `found` and an excursion is not.
    """
    n = 12
    excavator = {i: _blob(100, 60) for i in range(n)}
    # The bucket sits still, except at sample 5 where a barely-believed mask puts
    # it somewhere absurd.
    bucket = {i: _blob(60, 40) for i in range(n)}
    bucket[5] = _blob(180, 110, half=4)

    confident = [0.9] * n
    doubtful = [0.9] * n
    doubtful[5] = 0.10  # below the 0.35 floor

    kept, _ = build_features(
        _confident_result(n, confident), {"excavator": excavator, "bucket": bucket}, CONFIG
    )
    dropped, _ = build_features(
        _confident_result(n, doubtful), {"excavator": excavator, "bucket": bucket}, CONFIG
    )

    assert kept.found[5], "a confident mask must be kept"
    assert not dropped.found[5], "a mask below the floor must be reported as missing"
    # And the excursion must be gone from the signal, not merely flagged.
    assert abs(dropped.height[5] - dropped.height[4]) < abs(kept.height[5] - kept.height[4])


def test_the_confidence_floor_comes_from_the_config():
    """A different floor must actually change the outcome, or the knob is a lie."""
    import dataclasses

    n = 12
    excavator = {i: _blob(100, 60) for i in range(n)}
    bucket = {i: _blob(60, 40) for i in range(n)}
    confidences = [0.9] * n
    confidences[5] = 0.50  # above the default floor, below a raised one

    default = build_features(
        _confident_result(n, confidences), {"excavator": excavator, "bucket": bucket}, CONFIG
    )[0]
    strict_config = dataclasses.replace(
        CONFIG, features=dataclasses.replace(CONFIG.features, min_sample_confidence=0.70)
    )
    strict = build_features(
        _confident_result(n, confidences),
        {"excavator": excavator, "bucket": bucket},
        strict_config,
    )[0]

    assert default.found[5], "0.50 clears the 0.35 default"
    assert not strict.found[5], "0.50 does not clear a 0.70 floor"
