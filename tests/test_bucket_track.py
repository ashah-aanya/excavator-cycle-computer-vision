"""Tests for tracking the bucket in a session of its own.

The bucket used to be SAM 2's second object inside one shared session. It is now
a second **session** carrying one object, because the model's forward loop
couples objects to a shared frame cursor: an object flagged as pending on a
frame where it has no prompt is run as a conditioning frame with no prompt at
all, and stores a memory built from nothing. Sharing a session therefore forced
both prompts onto the detector's frame -- sample 290 of 296 on the development
video -- and that frame is chosen for detection confidence, not for where the
bucket is.

No model runs here. What is tested is everything around the model, which is
where this goes wrong *silently*:

* the bucket's seed frame, which comes from ``seeding.choose_seed``'s ranking
  over the whole clip and must be free to differ from the excavator's;
* the body core, which must be computed from every mask in the run rather than
  from a window around someone else's seed;
* one object per session -- a session that reports two has been reused, and the
  masks it returns belong to a memory bank built for something else;
* a mask file that holds two mask sets, and still reads the several
  single-object files already sitting in ``outputs/`` and ``CACHE/``;
* the output split -- a two-object prediction is ``(2, 1, H, W)`` and the old
  code read row 0 and called it the answer. One object per session makes that
  unreachable today; the guard stays so that re-merging the sessions cannot
  reintroduce it unnoticed;
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
from excavator_cycles import track as track_stage
from excavator_cycles.config import Config
from excavator_cycles.kinematics import body_core
from excavator_cycles.seeding import BucketSeed
from excavator_cycles.track import (
    BUCKET_OBJECT_ID,
    EXCAVATOR_OBJECT_ID,
    bucket_prompt_payload,
    derive_bucket_seed,
    evaluate_bucket_quality,
    load_result,
    register_prompt,
    split_objects,
    track_object,
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


# --- one object per session, streamed ------------------------------------------


class StubProcessor:
    """A processor that records registrations and assigns, as the real one does.

    ``add_inputs_to_inference_session`` ends with
    ``inference_session.obj_with_new_inputs = obj_ids`` -- not an append. That
    single line is why two objects in one session needed the pending list
    restoring afterwards, and why one object per session needs nothing.

    Calling it prepares ONE frame, as streaming does: the fake passes the frame
    through untouched so the model can report which sample it was given.
    """

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, **prompt):
        ids = [obj_ids] if isinstance(obj_ids, int) else list(obj_ids)
        inference_session.calls.append((frame_idx, ids, prompt))
        inference_session.obj_with_new_inputs = ids

    def __call__(self, images, device, return_tensors):
        return SimpleNamespace(pixel_values=[images], original_sizes=[images.shape[:2]])


def stub_processor():
    return StubProcessor()


class FakeFrame:
    """Stands in for one RGB frame, and remembers which sample it is."""

    def __init__(self, sample: int):
        self.sample = sample
        self.shape = (*SHAPE, 3)


def fake_frames(count: int) -> list[FakeFrame]:
    return [FakeFrame(i) for i in range(count)]


def stub_session(index: int = 0):
    """The parts of a SAM 2 session the tracker reads and prunes, plus a record of
    what it was fed, the most it ever held at once, and any history SAM 2 would
    have looked for and not found."""
    return SimpleNamespace(
        index=index,
        missing_history=[],
        obj_with_new_inputs=[],
        calls=[],
        inference_device="cpu",
        processed_frames={},
        output_dict_per_obj={0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
        fed=[],
        peak_frames=0,
        peak_outputs=0,
    )


def session_factory():
    """A ``new_session`` that keeps every session it made, in order."""
    made: list = []

    def new_session():
        made.append(stub_session(index=len(made)))
        return made[-1]

    new_session.made = made
    return new_session


def stub_model():
    """A model that stores what the real one stores, so a test can see what a
    session holds.

    Real SAM 2 keeps every frame it is given in ``processed_frames`` and every
    frame's output in ``output_dict_per_obj`` -- the prompted frame under
    ``cond_frame_outputs``, the rest under ``non_cond_frame_outputs``. Both would
    grow with the clip's length if nothing removed them, which is exactly what
    the bound test watches. ``frame`` is a required argument, as it must be: a
    call without it would take SAM 2 out of streaming mode.

    It also checks the opposite failure. At step ``s`` real SAM 2 reads the
    outputs of steps ``s-1 .. s-15`` (object pointers) and ``s-1 .. s-6`` (mask
    memory); pruning any of those does not crash -- it silently tracks from a
    thinner memory. So every call records which of them are absent.
    """
    reach = 16  # max_object_pointers_in_encoder: look back reach - 1 steps

    def model(inference_session, frame_idx, frame):
        session = inference_session
        held = session.output_dict_per_obj[0]["non_cond_frame_outputs"]
        wanted = range(max(1, frame_idx - (reach - 1)), frame_idx)
        session.missing_history += [(frame_idx, k) for k in wanted if k not in held]
        session.processed_frames[frame_idx] = frame
        key = "cond_frame_outputs" if frame_idx == 0 else "non_cond_frame_outputs"
        session.output_dict_per_obj[0][key][frame_idx] = frame.sample
        session.fed.append(frame.sample)
        session.peak_frames = max(session.peak_frames, len(session.processed_frames))
        held = len(session.output_dict_per_obj[0]["non_cond_frame_outputs"])
        session.peak_outputs = max(session.peak_outputs, held)
        return SimpleNamespace(frame_idx=frame_idx, sample=frame.sample, session=session.index)

    model.config = SimpleNamespace(num_maskmem=7, max_object_pointers_in_encoder=reach)
    return model


def one_object(obj_id: int, mask=None, empty_at=()):
    """An ``unpack`` that reports a single object, as a real session must."""
    band = machine() if mask is None else mask

    def unpack(output):
        found = np.zeros(SHAPE, bool) if output.sample in empty_at else band
        return {obj_id: (found, 0.9)}

    return unpack


BOX = {"input_boxes": [[[1.0, 2.0, 3.0, 4.0]]]}


def test_a_session_is_prompted_for_exactly_one_object_on_one_frame():
    """The registration a shared session could not make: one call, one id.

    Two objects in one session had to name the same frame, because ``forward``
    looks up an object's prompt for the frame it is processing and finds nothing
    when the object was prompted elsewhere. One object per session removes the
    constraint entirely, so there is exactly one call to check.
    """
    session = stub_session()
    register_prompt(
        stub_processor(), session, 53, BUCKET_OBJECT_ID, {"input_masks": [machine()]}
    )

    assert len(session.calls) == 1
    frame_idx, ids, prompt = session.calls[0]
    assert frame_idx == 53
    assert ids == [BUCKET_OBJECT_ID]
    assert list(prompt) == ["input_masks"]
    assert session.obj_with_new_inputs == [BUCKET_OBJECT_ID]


def test_tracking_streams_forward_then_backward_in_two_sessions():
    """One seed, both directions -- which is what lets the seed be the best frame.

    Each direction is a stream of its own, fed seed-first: forward runs to the
    end, backward runs to the start, and each session is prompted exactly once,
    on the first frame it sees.
    """
    new_session = session_factory()
    masks, confidences = track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(8),
        EXCAVATOR_OBJECT_ID,
        3,
        BOX,
        one_object(EXCAVATOR_OBJECT_ID),
    )

    forward, backward = new_session.made
    assert forward.fed == [3, 4, 5, 6, 7]
    assert backward.fed == [3, 2, 1, 0]
    for session in (forward, backward):
        assert [(f, ids) for f, ids, _ in session.calls] == [(0, [EXCAVATOR_OBJECT_ID])]
    assert sorted(masks) == list(range(8)), "forward and backward must both run"
    assert sorted(confidences) == list(range(8))


def test_a_box_prompt_is_sent_with_the_frame_size():
    """A streaming session holds no video, so SAM 2 cannot read the frame size
    from it -- and it refuses a box or point prompt without one."""
    new_session = session_factory()
    track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(4),
        EXCAVATOR_OBJECT_ID,
        0,
        BOX,
        one_object(EXCAVATOR_OBJECT_ID),
    )
    ((_, _, prompt),) = new_session.made[0].calls
    assert prompt["original_size"] == SHAPE


def test_a_seed_on_the_first_frame_needs_no_backward_stream():
    new_session = session_factory()
    track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(5),
        EXCAVATOR_OBJECT_ID,
        0,
        BOX,
        one_object(EXCAVATOR_OBJECT_ID),
    )
    assert len(new_session.made) == 1
    assert new_session.made[0].fed == [0, 1, 2, 3, 4]


def test_the_bucket_session_is_conditioned_on_the_bucket_seeds_own_frame():
    """The point of the second session: a frame the bucket ranking chose."""
    new_session = session_factory()
    track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(12),
        BUCKET_OBJECT_ID,
        5,
        {"input_masks": [machine()]},
        one_object(BUCKET_OBJECT_ID),
    )
    for session in new_session.made:
        assert session.fed[0] == 5, "every stream starts on the seed"
        assert [f for f, _, _ in session.calls] == [0], "prompted on its first frame"


def test_a_long_clip_holds_one_frame_and_a_fixed_window_of_outputs():
    """The reason for streaming, pinned: memory must not grow with the clip.

    Preparing every frame up front asked for 9.75 GiB in one allocation on an
    83 s clip at 10 Hz, and even with frames streamed SAM 2 keeps each frame's
    output (about 5.6 MiB) unless something removes it. So across a long clip a
    session may hold ONE prepared frame and at most ``output_window`` outputs --
    and the prompted frame's output, which every later frame attends to, must
    still be there at the end.
    """
    model = stub_model()
    new_session = session_factory()
    track_object(
        model,
        stub_processor(),
        new_session,
        fake_frames(500),
        EXCAVATOR_OBJECT_ID,
        250,
        BOX,
        one_object(EXCAVATOR_OBJECT_ID),
    )

    window = track_stage.output_window(model)
    assert len(new_session.made) == 2
    for session in new_session.made:
        assert len(session.fed) > window, "the stream must outlast the window to test it"
        assert session.peak_frames == 1, f"held {session.peak_frames} frames at once"
        assert session.peak_outputs <= window, f"held {session.peak_outputs} outputs"
        assert session.output_dict_per_obj[0]["cond_frame_outputs"] == {0: 250}
        assert session.missing_history == [], (
            f"pruned outputs SAM 2 still reads: {session.missing_history[:5]}"
        )


def test_the_seed_mask_comes_from_the_forward_stream():
    """Both streams predict the prompted seed frame, from identical inputs. One
    must win, and it is the forward one -- stated, and pinned, so the choice
    cannot flip unnoticed. Confidence here carries which session answered."""

    def by_session(output):
        return {EXCAVATOR_OBJECT_ID: (machine(), float(output.session))}

    _, confidences = track_object(
        stub_model(),
        stub_processor(),
        session_factory(),
        fake_frames(8),
        EXCAVATOR_OBJECT_ID,
        3,
        BOX,
        by_session,
    )
    assert confidences[3] == 0.0, "the seed must be taken from the forward stream"
    assert confidences[2] == 1.0 and confidences[4] == 0.0


def test_the_output_window_is_twice_sam2s_longest_look_back():
    """SAM 2 reads back 6 frames of mask memory and 15 of object pointers (sam2.1:
    num_maskmem 7, max_object_pointers_in_encoder 16). The window keeps twice the
    longer, read from the model's own config so a longer-memory checkpoint is
    handled rather than silently truncated."""
    sam21 = SimpleNamespace(
        config=SimpleNamespace(num_maskmem=7, max_object_pointers_in_encoder=16)
    )
    longer = SimpleNamespace(
        config=SimpleNamespace(num_maskmem=40, max_object_pointers_in_encoder=16)
    )
    assert track_stage.output_window(sam21) == 32
    assert track_stage.output_window(longer) == 80
    assert track_stage.output_window(SimpleNamespace()) == 32, "sam2.1's values when unknown"


def test_a_session_that_reports_a_second_object_is_refused():
    """A session carrying two objects has been reused, and its memory bank is
    conditioned on something the caller did not ask for. Failing loudly here is
    the difference between a wrong answer and no answer."""

    def two_objects(output):
        return {EXCAVATOR_OBJECT_ID: (machine(), 0.9), BUCKET_OBJECT_ID: (machine(), 0.5)}

    with pytest.raises(RuntimeError, match="exactly one object"):
        track_object(
            stub_model(),
            stub_processor(),
            session_factory(),
            fake_frames(4),
            EXCAVATOR_OBJECT_ID,
            0,
            BOX,
            two_objects,
        )


def test_an_empty_bucket_prediction_is_not_stored_but_is_still_scored():
    """SAM predicts on every frame whether or not it still believes the object is
    there, so keeping the empty masks would make bucket coverage 100% by
    construction. The excavator keeps its empty frames: the QA gate has to see
    them."""
    bucket, _ = track_object(
        stub_model(),
        stub_processor(),
        session_factory(),
        fake_frames(6),
        BUCKET_OBJECT_ID,
        0,
        {"input_masks": [machine()]},
        one_object(BUCKET_OBJECT_ID, empty_at=(2, 4)),
        keep_empty=False,
    )
    assert sorted(bucket) == [0, 1, 3, 5]

    excavator, confidences = track_object(
        stub_model(),
        stub_processor(),
        session_factory(),
        fake_frames(6),
        EXCAVATOR_OBJECT_ID,
        0,
        BOX,
        one_object(EXCAVATOR_OBJECT_ID, empty_at=(2, 4)),
        keep_empty=True,
    )
    assert sorted(excavator) == list(range(6))
    assert sorted(confidences) == list(range(6)), "confidence is recorded either way"


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


# --- deriving the seed from the excavator's whole mask set -------------------


# (elbow, bucket) through one working cycle. Deliberately spread rather than
# nudged frame by frame: a set in which the arm barely moves is one whose
# "persistent body" still contains most of the arm, and the pivot then lands
# part way up the boom -- the real failure mode the full mask set exists to
# avoid, not a fixture detail.
POSES = [
    ((150, 90), (240, 150)),
    ((120, 30), (200, 60)),
    ((100, 120), (150, 220)),
    ((140, 60), (260, 110)),
    ((110, 150), (120, 230)),
    ((160, 100), (280, 170)),
    ((130, 40), (230, 40)),
    ((105, 160), (100, 235)),
]


def sweep(start: int = 40) -> dict[int, np.ndarray]:
    """Masks in which the arm actually moves, keyed by sample ordinal."""
    return {start + i: machine(elbow=e, bucket=b) for i, (e, b) in enumerate(POSES)}


def clip(length: int = 13) -> dict[int, np.ndarray]:
    """A whole clip of excavator masks, with the arm at full stretch mid-way.

    The cycle repeats, so several frames carry a usable band and the ranking has
    something to choose between. Ordinal ``length // 2`` is the one the arm is
    fully extended on, which is what ``score_frame`` rewards -- reach, a compact
    band and mid-clip-ness all at once.
    """
    poses = list(sweep().values())
    masks = {i: poses[i % len(poses)] for i in range(length)}
    masks[length // 2] = machine(elbow=(160, 110), bucket=(300, 175))
    return masks


def test_the_bucket_seed_frame_is_ranked_not_inherited_from_the_excavator():
    """The whole reason the bucket gets a session of its own.

    Sharing one forced the bucket onto the detector's frame -- sample 290 of 296
    on the development video, six frames from the end and chosen for detection
    confidence. Here the excavator is seeded on the last ordinal and the bucket
    must still land on the frame the ranking prefers.
    """
    config = Config.load()
    masks = clip()
    excavator_seed = max(masks)  # what the detector picked: the end of the clip

    seed = derive_bucket_seed(masks, len(masks), None, config)

    assert seed is not None
    assert seed.sample != excavator_seed, "the bucket must not inherit that frame"
    assert seed.sample == len(masks) // 2, "the extended, mid-clip frame wins"


def test_every_masked_sample_is_scored_not_only_the_excavators_seed_frame():
    """A shared session left one real candidate and a list padded with None, so
    `choose_seed`'s ranking had nothing to rank and the answer could not depend
    on any other frame. Rearranging which ordinal carries which pose must now
    change which ordinal is chosen."""
    config = Config.load()
    masks = clip()

    first = derive_bucket_seed(masks, len(masks), None, config)

    moved = dict(masks)
    moved[2], moved[len(masks) // 2] = moved[len(masks) // 2], moved[2]
    second = derive_bucket_seed(moved, len(moved), None, config)

    assert first is not None and second is not None
    assert second.sample != first.sample


def test_the_seed_sample_is_an_ordinal_into_the_clip_not_a_list_index():
    """Masks are keyed by sample ordinal and the run may be missing some, so the
    candidate list is rebuilt at full length rather than packed."""
    config = Config.load()
    seed = derive_bucket_seed(sweep(40), 300, None, config)

    assert seed is not None
    assert seed.sample >= 40, "a packed list would put the seed near 0"
    assert seed.sample in sweep(40)


def test_the_band_lands_on_the_bucket_of_the_frame_it_chose():
    config = Config.load()
    masks = sweep(0)
    seed = derive_bucket_seed(masks, len(masks), None, config)

    assert seed is not None
    bucket_x, bucket_y = POSES[seed.sample][1]
    ys, xs = np.nonzero(seed.band)
    offset = np.hypot(xs.mean() - bucket_x, ys.mean() - bucket_y)
    assert offset < 30, f"band centre is {offset:.0f}px from the bucket"


def test_the_body_core_is_computed_from_every_mask_in_the_run(monkeypatch):
    """Not from a window around the excavator's seed, which is what a shared
    session forced: `body_core` of one mask is that mask, so a short window
    leaves the arm in the core and the pivot lands part way up the boom."""
    config = Config.load()
    masks = clip()
    seen: list[int] = []
    real = track_stage.body_core

    def spy(mask_list, quantile):
        seen.append(len(mask_list))
        return real(mask_list, quantile)

    monkeypatch.setattr(track_stage, "body_core", spy)
    seed = derive_bucket_seed(masks, len(masks), None, config)

    assert seed is not None
    assert seen == [len(masks)], f"body_core saw {seen} masks, not all {len(masks)}"


def test_no_usable_excavator_mask_yields_no_bucket_seed():
    """The stage then tracks the excavator alone rather than guessing."""
    config = Config.load()
    empty = np.zeros(SHAPE, bool)
    assert derive_bucket_seed({7: empty}, 300, None, config) is None
    assert derive_bucket_seed({}, 300, None, config) is None


def test_the_body_core_needs_several_frames_to_separate_the_arm():
    """Why the seed is derived after a full pass rather than during one.

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
    assert mask_io.load_objects(tmp_path / "masks.npz")[0].get("bucket", {}) == {}


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
    assert sorted(mask_io.load_objects(tmp_path / "masks.npz")[0]["bucket"]) == [1]
