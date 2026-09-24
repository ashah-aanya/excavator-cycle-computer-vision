"""Tests for tracking the bucket as SAM 2's second object.

No model runs here. What is tested is everything around the model, which is
where a second object goes wrong *silently*:

* a mask file that holds two mask sets, and still reads the several
  single-object files already sitting in ``outputs/`` and ``CACHE/``;
* the output split -- a two-object prediction is ``(2, 1, H, W)`` and the old
  code read row 0 and called it the answer, so the bucket would have vanished
  without an error;
* prompt registration -- the processor *assigns* its "has new inputs" list, so
  registering the second object erases the first's prompt and the excavator is
  then tracked from a memory bank it never built;
* prompt form -- SAM 2's own ablation prices a mask at 77.6 J&F against 72.9 for
  a box, so which form is sent is a real decision and its payload shape is
  exactly the sort of thing that fails only on a GPU an hour in;
* the new record fields, against a ``track.json`` written before they existed.

Everything downstream is keyed by **sample ordinal**, never by source frame
index, and the fixtures here use non-contiguous ordinals on purpose so that a
confusion between the two cannot pass.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from excavator_cycles import masks as mask_io
from excavator_cycles.config import Config
from excavator_cycles.kinematics import body_core
from excavator_cycles.seeding import BucketSeed
from excavator_cycles.track import (
    BUCKET_OBJECT_ID,
    EXCAVATOR_OBJECT_ID,
    bucket_prompt_payload,
    derive_bucket_seed,
    evaluate_bucket_quality,
    load_bucket_masks,
    load_result,
    register_prompts,
    split_objects,
)

SHAPE = (240, 320)
PIVOT = (60.0, 180.0)


def machine(elbow=(150, 90), bucket=(240, 150), bucket_radius=18):
    """A body blob on the pivot, a two-segment arm, and a blob for the bucket.

    The same shape as the seeding tests use, because the claim being checked
    here is that the stage hands ``seeding`` something it can work with -- not
    that the geodesic rule itself is right, which is tested there.
    """
    mask = np.zeros(SHAPE, bool)
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]

    mask |= np.hypot(xs - PIVOT[0], ys - PIVOT[1]) <= 34
    for start, end in ((PIVOT, elbow), (elbow, bucket)):
        steps = int(np.hypot(end[0] - start[0], end[1] - start[1])) * 2
        for t in np.linspace(0, 1, max(steps, 2)):
            cx = start[0] + t * (end[0] - start[0])
            cy = start[1] + t * (end[1] - start[1])
            mask |= np.hypot(xs - cx, ys - cy) <= 7
    mask |= np.hypot(xs - bucket[0], ys - bucket[1]) <= bucket_radius
    return mask


def blob(cx: int, cy: int, radius: int) -> np.ndarray:
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    return np.hypot(xs - cx, ys - cy) <= radius


# --- storing two mask sets in one file ---------------------------------------


def test_both_mask_sets_survive_one_round_trip(tmp_path):
    """Two objects, one file: a second `save` call would truncate the first."""
    excavator = {4: machine(), 9: machine(bucket=(200, 160))}
    bucket = {4: blob(240, 150, 18), 9: blob(200, 160, 18)}

    path = tmp_path / "masks.npz"
    mask_io.save_objects(path, {"excavator": excavator, "bucket": bucket}, SHAPE)

    objects, shape = mask_io.load_objects(path)
    assert shape == SHAPE
    assert sorted(objects) == ["bucket", "excavator"]
    assert sorted(objects["excavator"]) == [4, 9]
    assert sorted(objects["bucket"]) == [4, 9]
    assert np.array_equal(objects["excavator"][9], excavator[9])
    assert np.array_equal(objects["bucket"][9], bucket[9])


def test_a_file_without_a_bucket_still_loads_and_reports_no_bucket(tmp_path):
    """Several single-object runs are already on disk; none of them may break."""
    excavator = {0: machine(), 1: machine(bucket=(200, 160))}
    path = tmp_path / "masks.npz"
    mask_io.save(path, excavator, SHAPE)

    objects, shape = mask_io.load_objects(path)
    assert "bucket" not in objects, "a legacy file must report no bucket, not an empty one"
    assert sorted(objects["excavator"]) == [0, 1]

    loaded, shape = mask_io.load(path)
    assert shape == SHAPE
    assert np.array_equal(loaded[1], excavator[1])


def test_an_empty_bucket_set_is_distinguishable_from_no_bucket_set(tmp_path):
    """A run that tracked the bucket and lost it is not a run that never tried."""
    path = tmp_path / "masks.npz"
    mask_io.save_objects(path, {"excavator": {0: machine()}, "bucket": {}}, SHAPE)

    objects, _ = mask_io.load_objects(path)
    assert objects["bucket"] == {}


def test_an_unknown_mask_set_is_refused_rather_than_written(tmp_path):
    """Key prefixes are a fixed table; a typo would write a file nothing reads."""
    with pytest.raises(ValueError, match="unknown mask set"):
        mask_io.save_objects(tmp_path / "masks.npz", {"arm": {0: machine()}}, SHAPE)


# --- splitting a two-object prediction ---------------------------------------


def prediction(rows: list[np.ndarray]) -> np.ndarray:
    """What `post_process_masks` returns: (objects, 1, H, W), already binarised."""
    return np.stack([row[None, :, :] for row in rows])


def test_the_bucket_row_is_not_dropped_when_two_objects_are_tracked():
    """The failure this whole function exists to prevent: `processed[0, 0]`.

    With one object, row 0 and "the excavator" coincide. With two, reading row 0
    keeps the excavator and throws the bucket away without raising anything.
    """
    excavator, bucket = machine(), blob(240, 150, 18)
    found = split_objects(
        [EXCAVATOR_OBJECT_ID, BUCKET_OBJECT_ID],
        prediction([excavator, bucket]),
        np.array([2.0, 1.0]),
        {EXCAVATOR_OBJECT_ID: 0.05, BUCKET_OBJECT_ID: 0.01},
    )

    assert sorted(found) == [EXCAVATOR_OBJECT_ID, BUCKET_OBJECT_ID]
    assert np.array_equal(found[EXCAVATOR_OBJECT_ID][0], excavator)
    assert np.array_equal(found[BUCKET_OBJECT_ID][0], bucket)


def test_objects_are_read_by_reported_id_not_by_row_order():
    """Row order follows the session's registration order, not the ids."""
    excavator, bucket = machine(), blob(240, 150, 18)
    found = split_objects(
        [BUCKET_OBJECT_ID, EXCAVATOR_OBJECT_ID],  # bucket registered first
        prediction([bucket, excavator]),
        np.array([1.0, 2.0]),
        {EXCAVATOR_OBJECT_ID: 0.05, BUCKET_OBJECT_ID: 0.01},
    )
    assert np.array_equal(found[BUCKET_OBJECT_ID][0], bucket)
    assert np.array_equal(found[EXCAVATOR_OBJECT_ID][0], excavator)


def test_each_object_gets_its_own_speck_threshold():
    """The bucket mask is an order of magnitude smaller, so the same relative
    cut would erase fragments of it that are real."""
    small = blob(200, 120, 12)  # ~450 px
    fragment = np.zeros(SHAPE, bool)
    fragment[20:23, 20:24] = True  # 12 px, ~2.6% of the blob

    combined = small | fragment
    found = split_objects(
        [EXCAVATOR_OBJECT_ID, BUCKET_OBJECT_ID],
        prediction([combined, combined]),
        None,
        {EXCAVATOR_OBJECT_ID: 0.05, BUCKET_OBJECT_ID: 0.01},
    )
    assert not found[EXCAVATOR_OBJECT_ID][0][21, 21], "5% cuts the fragment"
    assert found[BUCKET_OBJECT_ID][0][21, 21], "1% keeps it"


def test_confidence_is_the_presence_logit_turned_into_a_probability():
    found = split_objects(
        [EXCAVATOR_OBJECT_ID, BUCKET_OBJECT_ID],
        prediction([machine(), blob(240, 150, 18)]),
        np.array([0.0, -4.0]),
        {},
    )
    assert found[EXCAVATOR_OBJECT_ID][1] == pytest.approx(0.5)
    assert found[BUCKET_OBJECT_ID][1] == pytest.approx(0.0179, abs=1e-3)


def test_a_missing_presence_logit_reads_as_unknown_not_as_zero():
    """nan says "not measured"; 0.0 would say "certainly absent"."""
    found = split_objects([EXCAVATOR_OBJECT_ID], prediction([machine()]), None, {})
    assert np.isnan(found[EXCAVATOR_OBJECT_ID][1])


# --- registering both prompts ------------------------------------------------


def stub_processor():
    """A processor that reproduces the one behaviour under test: assignment.

    ``add_inputs_to_inference_session`` ends with
    ``inference_session.obj_with_new_inputs = obj_ids`` -- not an append. That
    single line is why the second object silently cancels the first.
    """

    def add_inputs_to_inference_session(inference_session, frame_idx, obj_ids, **prompt):
        ids = [obj_ids] if isinstance(obj_ids, int) else list(obj_ids)
        inference_session.calls.append((frame_idx, ids, prompt))
        inference_session.obj_with_new_inputs = ids

    return SimpleNamespace(add_inputs_to_inference_session=add_inputs_to_inference_session)


def test_registering_the_bucket_does_not_erase_the_excavators_pending_prompt():
    """Both objects must still be pending when the first forward pass runs.

    An object missing from this list is treated as having no new inputs, so its
    prompt is never read and it is tracked from an empty memory bank.
    """
    session = SimpleNamespace(obj_with_new_inputs=[], calls=[])
    register_prompts(
        stub_processor(),
        session,
        53,
        {
            EXCAVATOR_OBJECT_ID: {"input_boxes": [[[1.0, 2.0, 3.0, 4.0]]]},
            BUCKET_OBJECT_ID: {"input_masks": [np.zeros(SHAPE, bool)]},
        },
    )
    assert sorted(session.obj_with_new_inputs) == [EXCAVATOR_OBJECT_ID, BUCKET_OBJECT_ID]


def test_each_object_is_registered_in_its_own_call_on_the_same_frame():
    """Separate calls dodge the padding rule; the same frame is what forward needs.

    Points for two objects in one call must have equal point counts, and these
    do not. Different conditioning frames are worse: the object whose prompt is
    elsewhere is conditioned on nothing at all.
    """
    session = SimpleNamespace(obj_with_new_inputs=[], calls=[])
    register_prompts(
        stub_processor(),
        session,
        53,
        {
            EXCAVATOR_OBJECT_ID: {"input_boxes": [[[1.0, 2.0, 3.0, 4.0]]]},
            BUCKET_OBJECT_ID: {"input_masks": [np.zeros(SHAPE, bool)]},
        },
    )
    assert len(session.calls) == 2
    assert [frame for frame, _, _ in session.calls] == [53, 53]
    assert [ids for _, ids, _ in session.calls] == [[EXCAVATOR_OBJECT_ID], [BUCKET_OBJECT_ID]]


# --- the three prompt forms --------------------------------------------------


def a_seed() -> BucketSeed:
    band = blob(240, 150, 18)
    return BucketSeed(
        sample=53,
        points=np.array([[240.0, 150.0], [244.0, 152.0]]),
        negatives=np.array([[180.0, 120.0]]),
        box=(222, 132, 259, 169),
        band=band,
        score=2.5,
        reach=88.0,
    )


def test_the_mask_form_sends_the_band_itself():
    """77.6 J&F against 72.9 for a box -- and the band is already computed."""
    payload = bucket_prompt_payload(a_seed(), "mask")
    assert list(payload) == ["input_masks"]
    assert len(payload["input_masks"]) == 1, "one mask per object id"
    assert payload["input_masks"][0].shape == SHAPE
    assert payload["input_masks"][0].dtype == bool


def test_the_box_form_sends_one_box_nested_per_batch_and_object():
    payload = bucket_prompt_payload(a_seed(), "box")
    assert payload == {"input_boxes": [[[222.0, 132.0, 259.0, 169.0]]]}


def test_the_points_form_labels_positives_one_and_negatives_zero():
    """Negatives on the stick are what stop SAM claiming the whole arm."""
    payload = bucket_prompt_payload(a_seed(), "points")
    assert sorted(payload) == ["input_labels", "input_points"]

    points = payload["input_points"][0][0]
    labels = payload["input_labels"][0][0]
    assert len(points) == len(labels) == 3
    assert labels == [1, 1, 0]
    assert points[0] == [240.0, 150.0]
    assert points[-1] == [180.0, 120.0], "the negative comes last, matching its label"


def test_an_unknown_prompt_form_is_refused():
    with pytest.raises(ValueError, match="unknown bucket prompt form"):
        bucket_prompt_payload(a_seed(), "scribble")


# --- deriving the seed from the first pass -----------------------------------


def sweep(start: int = 40) -> dict[int, np.ndarray]:
    """A window of masks in which the arm actually moves, keyed by sample ordinal.

    The poses are deliberately spread across a working cycle rather than nudged
    frame by frame. A window in which the arm barely moves is a window whose
    "persistent body" still contains most of the arm, and the pivot then lands
    part way up the boom -- which is the real failure mode this window exists to
    avoid, not a fixture detail.
    """
    poses = [
        ((150, 90), (240, 150)),
        ((120, 30), (200, 60)),
        ((100, 120), (150, 220)),
        ((140, 60), (260, 110)),
        ((110, 150), (120, 230)),
        ((160, 100), (280, 170)),
        ((130, 40), (230, 40)),
        ((105, 160), (100, 235)),
    ]
    return {start + i: machine(elbow=e, bucket=b) for i, (e, b) in enumerate(poses)}


def test_the_bucket_seed_is_taken_on_the_excavator_seed_sample():
    """Both objects have to be conditioned on the same frame, so the bucket is
    seeded there and nowhere else -- and its `sample` is a sample ordinal."""
    config = Config.load()
    seed = derive_bucket_seed(sweep(40), 300, 40, None, config)

    assert seed is not None
    assert seed.sample == 40, "the seed must carry the ordinal, not a list index"
    ys, xs = np.nonzero(seed.band)
    offset = np.hypot(xs.mean() - 240, ys.mean() - 150)
    assert offset < 30, f"band centre is {offset:.0f}px from the bucket"


def test_no_usable_excavator_mask_yields_no_bucket_seed():
    """The stage then tracks the excavator alone rather than guessing."""
    config = Config.load()
    empty = np.zeros(SHAPE, bool)
    assert derive_bucket_seed({7: empty}, 300, 7, None, config) is None
    assert derive_bucket_seed({}, 300, 7, None, config) is None


def test_the_body_core_needs_several_frames_to_separate_the_arm():
    """Why the first pass propagates a window instead of one forward pass.

    The geodesic seed works by removing the persistent body before tracing the
    arm. Over one frame "persistent" means everything, so the subtraction
    removes the whole mask and the tracks are left competing with the bucket for
    farthest-along-the-metal.
    """
    quantile = Config.load().geometry.occupancy_quantile
    single = machine()

    assert np.array_equal(body_core([single], quantile), single)

    window = list(sweep().values())
    core = body_core(window, quantile)
    assert core.sum() < 0.6 * window[0].sum(), "the arm must drop out of a real core"
    assert core[round(PIVOT[1]), round(PIVOT[0])], "the body must stay in it"


# --- quality, reported rather than gated -------------------------------------


def test_bucket_quality_reports_coverage_containment_and_area_stability():
    config = Config.load()
    excavator = {i: machine() for i in range(10)}
    bucket = {i: blob(240, 150, 18) for i in range(9)}

    qa = evaluate_bucket_quality(excavator, bucket, 10, config)
    assert qa["coverage"] == pytest.approx(0.9)
    assert qa["samples_with_mask"] == 9
    assert qa["containment_median"] == pytest.approx(1.0)
    assert qa["area_p90_over_p10"] == pytest.approx(1.0)
    assert qa["notes"] == []


def test_a_bucket_that_wanders_off_the_machine_shows_up_as_low_containment():
    """Area and confidence would both still look healthy; only this catches it."""
    config = Config.load()
    excavator = {i: machine() for i in range(10)}
    bucket = {i: blob(40, 40, 18) for i in range(10)}  # on the spoil pile, not the arm

    qa = evaluate_bucket_quality(excavator, bucket, 10, config)
    assert qa["containment_median"] == pytest.approx(0.0)
    assert any("outside the machine" in note for note in qa["notes"])


def test_a_bucket_that_swells_up_the_arm_shows_up_as_unstable_area():
    config = Config.load()
    excavator = {i: machine() for i in range(10)}
    bucket = {i: blob(240, 150, 10 if i < 5 else 40) for i in range(10)}

    qa = evaluate_bucket_quality(excavator, bucket, 10, config)
    assert qa["area_p90_over_p10"] > config.qa.max_bucket_area_ratio
    assert any("varies by" in note for note in qa["notes"])


def test_bucket_quality_survives_a_run_that_never_found_the_bucket():
    config = Config.load()
    qa = evaluate_bucket_quality({0: machine()}, {}, 10, config)
    assert qa["coverage"] == 0.0
    assert qa["containment_median"] is None
    assert qa["area_p90_over_p10"] is None
    assert any("bucket mask" in note for note in qa["notes"])


# --- reading a run written before the bucket existed -------------------------


def legacy_track_json() -> dict:
    """A `track.json` exactly as it was written before this change.

    Constructed here rather than copied from `outputs/`, so the claim keeps
    holding after those files are regenerated.
    """
    return {
        "video": "clip.mp4",
        "video_sha256": "0" * 64,
        "config_digest": "1" * 16,
        "width": 480,
        "height": 272,
        "fps": 30.0,
        "duration_seconds": 5.8,
        "rate_hz": 10.0,
        "seed": {"sample_position": 2, "frame_index": 6, "box": [1, 2, 3, 4]},
        "frames": [
            {
                "frame_index": i * 3,
                "time_seconds": i * 0.1,
                "mask_area_fraction": 0.07,
                "sam_confidence": 0.9,
                "has_mask": True,
            }
            for i in range(4)
        ],
        "qa": {"status": "pass", "failures": []},
    }


def test_an_old_track_json_still_loads_without_the_new_fields(tmp_path):
    """`load_result` rebuilds these dataclasses straight from JSON, so a field
    without a default would make every completed run unreadable."""
    (tmp_path / "track.json").write_text(json.dumps(legacy_track_json()))
    mask_io.save(tmp_path / "masks.npz", {i: machine() for i in range(4)}, SHAPE)

    result, masks = load_result(tmp_path)

    assert result.bucket_seed is None
    assert len(masks) == 4
    assert result.frames[0].has_bucket_mask is False
    assert result.frames[0].bucket_area_fraction == 0.0
    assert result.frames[0].bucket_confidence == 0.0
    assert load_bucket_masks(tmp_path) == {}


def test_a_run_with_a_bucket_round_trips_through_the_same_reader(tmp_path):
    data = legacy_track_json()
    data["bucket_seed"] = {"sample_position": 2, "prompt": "mask", "score": 2.5}
    data["frames"][1].update(
        {"bucket_area_fraction": 0.009, "bucket_confidence": 0.8, "has_bucket_mask": True}
    )
    (tmp_path / "track.json").write_text(json.dumps(data))
    mask_io.save_objects(
        tmp_path / "masks.npz",
        {"excavator": {i: machine() for i in range(4)}, "bucket": {1: blob(240, 150, 18)}},
        SHAPE,
    )

    result, masks = load_result(tmp_path)
    assert result.bucket_seed["prompt"] == "mask"
    assert result.frames[1].has_bucket_mask is True
    assert len(masks) == 4, "the excavator masks must be unaffected by the second set"
    assert sorted(load_bucket_masks(tmp_path)) == [1]
