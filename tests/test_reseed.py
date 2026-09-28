"""Tests for finding the bucket again after SAM 2 loses it (reseed.py).

The failure this exists for: on random.mp4 the bucket went into the truck bed at
34.7 s and SAM 2 never claimed it again -- 10.6 s with no bucket while it was
plainly in view. No model runs here; what is tested is the audit that finds a
lost span, the choice of where to reseed inside it, and the bookkeeping that
must touch that span and nothing else:

* a loss shorter than ``bucket_lost_seconds`` is a blurred frame, not a loss;
* "missing" is the features stage's own floor, and a NaN confidence is unknown;
* a reseed replaces the span's samples and leaves every other sample alone;
* a span with no usable frame is recorded as a gap and never retried in a loop;
* the ranking picks inside the span, and never over the truck;
* a limited SAM run asks the model for exactly the span's two legs.
"""

from __future__ import annotations

import numpy as np

from excavator_cycles import seeding
from excavator_cycles.reseed import (
    LostSpan,
    lost_spans,
    missing_samples,
    reseed_lost_spans,
)
from excavator_cycles.seeding import BucketSeed, choose_seed
from excavator_cycles.track import BUCKET_OBJECT_ID, track_object
from test_bucket_track import fake_frames, session_factory, stub_model, stub_processor

RATE = 10.0
FLOOR = 0.35
SHAPE = (240, 320)
PIVOT = (60.0, 180.0)
SCALE = 120.0


def times(count: int) -> np.ndarray:
    return np.arange(count) / RATE


def blob() -> np.ndarray:
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    return mask


def bucket_track(count: int, lost: range = range(0)):
    """A bucket seen with confidence 0.95 everywhere except ``lost``."""
    masks = {i: blob() for i in range(count) if i not in lost}
    confidences = {i: (0.08 if i in lost else 0.95) for i in range(count)}
    return masks, confidences


# --- the audit ---------------------------------------------------------------


def test_a_sample_is_missing_without_a_mask_or_below_the_floor():
    masks, confidences = bucket_track(6)
    del masks[1]  # no mask
    confidences[2] = 0.2  # a mask SAM itself doubts
    masks[3] = np.zeros((8, 8), bool)  # an empty mask is no mask
    confidences[4] = float("nan")  # no confidence reported: unknown, not doubt

    missing = missing_samples(6, masks, confidences, FLOOR)
    assert missing.tolist() == [False, True, True, True, False, False]


def test_a_blurred_moment_is_not_a_loss():
    """Nine missing samples at 10 Hz is 0.9 s -- under the second it must last."""
    missing = np.zeros(40, bool)
    missing[10:19] = True
    assert lost_spans(missing, times(40), 1.0) == []


def test_a_second_of_missing_samples_is_a_lost_span():
    """Ten samples at 10 Hz last a full second: first to last plus one spacing."""
    missing = np.zeros(40, bool)
    missing[10:20] = True
    assert lost_spans(missing, times(40), 1.0) == [LostSpan(10, 19)]


def test_a_loss_that_runs_to_the_end_of_the_clip_is_found():
    """random.mp4's shape: lost at 34.7 s and never found again."""
    missing = np.zeros(50, bool)
    missing[35:] = True
    assert lost_spans(missing, times(50), 1.0) == [LostSpan(35, 49)]


# --- reseeding ---------------------------------------------------------------


def fake_seed(sample: int) -> BucketSeed:
    return BucketSeed(
        sample=sample,
        points=np.zeros((1, 2)),
        negatives=np.zeros((0, 2)),
        box=(0, 0, 1, 1),
        band=blob(),
        score=1.0,
        reach=1.0,
    )


def run(count, masks, confidences, find_seed, track_span, max_reseeds=5):
    return reseed_lost_spans(
        count,
        times(count),
        masks,
        confidences,
        floor=FLOOR,
        min_seconds=1.0,
        max_reseeds=max_reseeds,
        find_seed=find_seed,
        track_span=track_span,
    )


def recovering_tracker(calls):
    """A SAM stand-in that finds the bucket on every sample of the span it is given."""

    def track_span(seed, span):
        calls.append((seed.sample, span))
        found = range(span.start, span.end + 1)
        return {i: blob() for i in found}, {i: 0.9 for i in found}

    return track_span


def test_a_lost_span_is_reseeded_from_a_frame_inside_it():
    masks, confidences = bucket_track(50, lost=range(30, 50))
    asked, calls = [], []

    def find_seed(positions):
        asked.append(positions)
        return fake_seed(40)

    records = run(50, masks, confidences, find_seed, recovering_tracker(calls))

    assert asked == [range(30, 50)], "the ranking must search the lost span only"
    assert calls == [(40, LostSpan(30, 49))]
    assert not missing_samples(50, masks, confidences, FLOOR).any()
    assert records == [
        {
            "start_seconds": 3.0,
            "end_seconds": 4.9,
            "missing_before": 20,
            "seed_sample": 40,
            "seed_seconds": 4.0,
            "missing_after": 0,
        }
    ]


def test_a_reseed_touches_the_lost_span_and_nothing_else():
    """The samples either side were tracked fine; a reseed must not rewrite them."""
    masks, confidences = bucket_track(50, lost=range(20, 32))
    before = {i: masks[i] for i in masks}
    sentinel = np.ones((8, 8), bool)

    def track_span(seed, span):
        # A tracker that (wrongly) reports samples outside the span as well.
        return {i: sentinel for i in range(50)}, {i: 0.5 for i in range(50)}

    run(50, masks, confidences, lambda positions: fake_seed(25), track_span)

    for i in range(50):
        if 20 <= i <= 31:
            assert masks[i] is sentinel
        else:
            assert masks[i] is before[i], f"sample {i} outside the span was rewritten"
            assert confidences[i] == 0.95


def test_a_span_where_the_bucket_never_reappears_stays_a_gap():
    """No frame clear of the truck: no SAM pass, one record, and no retry loop."""
    masks, confidences = bucket_track(50, lost=range(30, 50))
    calls = []
    records = run(50, masks, confidences, lambda positions: None, recovering_tracker(calls))

    assert calls == []
    assert len(records) == 1
    assert records[0]["seed_sample"] is None
    assert missing_samples(50, masks, confidences, FLOOR)[30:].all()


def test_a_reseed_that_does_not_recover_is_not_retried():
    """The bucket stays buried after the reseed too. Once per span, then stop."""
    masks, confidences = bucket_track(50, lost=range(30, 50))
    calls = []

    def track_span(seed, span):
        calls.append(span)
        found = range(span.start, span.end + 1)
        return {}, {i: 0.05 for i in found}

    records = run(50, masks, confidences, lambda positions: fake_seed(40), track_span)
    assert len(calls) == 1
    assert records[0]["missing_after"] == 20


def test_every_separate_loss_gets_its_own_reseed():
    masks, confidences = bucket_track(80, lost=set(range(10, 25)) | set(range(50, 70)))
    calls = []
    run(
        80,
        masks,
        confidences,
        lambda positions: fake_seed(positions.start + 2),
        recovering_tracker(calls),
    )
    assert [span for _, span in calls] == [LostSpan(10, 24), LostSpan(50, 69)]


def test_the_number_of_reseeds_is_bounded():
    lost = set().union(*(range(start, start + 12) for start in range(0, 200, 20)))
    masks, confidences = bucket_track(200, lost=lost)
    records = run(200, masks, confidences, lambda p: None, recovering_tracker([]), 3)
    assert len(records) == 3, "ten separate losses, but only three attempts allowed"


def test_a_clean_run_is_never_reseeded():
    masks, confidences = bucket_track(50)
    records = run(50, masks, confidences, lambda p: fake_seed(0), recovering_tracker([]))
    assert records == []


# --- where the reseed lands ----------------------------------------------------


def machine(bucket: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A body at the pivot, one straight arm, and a bucket blob at its end."""
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    body = np.hypot(xs - PIVOT[0], ys - PIVOT[1]) <= 34
    mask = body.copy()
    for t in np.linspace(0, 1, 400):
        cx = PIVOT[0] + t * (bucket[0] - PIVOT[0])
        cy = PIVOT[1] + t * (bucket[1] - PIVOT[1])
        mask |= np.hypot(xs - cx, ys - cy) <= 7
    mask |= np.hypot(xs - bucket[0], ys - bucket[1]) <= 18
    return mask, body


def test_the_reseed_is_chosen_inside_the_span_only():
    core = machine((200, 150))[1]
    # The best frame of the clip (longest reach) is outside the span.
    masks = [machine((200, 150))[0] for _ in range(10)]
    masks[1] = machine((280, 150))[0]
    seed = choose_seed(masks, core, PIVOT, SCALE, positions=range(5, 10))
    assert seed is not None
    assert 5 <= seed.sample < 10


def test_a_bucket_over_the_truck_is_never_the_reseed():
    """Over the bed is where the bucket hides; a band there is the arm reaching in."""
    core = machine((200, 150))[1]
    truck = (230.0, 100.0, 310.0, 200.0)
    over, clear = machine((270, 150))[0], machine((170, 190))[0]
    masks = [over, over, clear, over]

    seed = choose_seed(masks, core, PIVOT, SCALE, truck_box=truck, clear_of_truck=True)
    assert seed is not None
    assert seed.sample == 2

    none_clear = choose_seed([over] * 4, core, PIVOT, SCALE, truck, clear_of_truck=True)
    assert none_clear is None


def test_a_band_with_any_piece_over_the_truck_is_not_clear_of_it(monkeypatch):
    """Seen on 15107628: one piece on the bucket, one on the cab inside the truck
    box. Their centre lay between the two, clear of the truck, so a centre test
    let the frame through and SAM 2 would have been prompted with both pieces."""
    truck = (230.0, 100.0, 310.0, 200.0)
    mask, core = machine((170, 190))
    bucket = np.zeros(SHAPE, bool)
    bucket[180:200, 160:180] = True  # on the bucket, clear of the truck
    split = bucket.copy()
    split[150:160, 300:310] = True  # a second piece, inside the truck box
    stick = np.zeros(SHAPE, bool)
    stick[185:190, 120:130] = True
    # Frame 0 has the split band and the longer reach, so it wins on score.
    bands = {0: (split, stick, 400.0), 1: (bucket, stick, 100.0)}
    frames = [mask.copy(), mask.copy()]
    frames[0][0, 0] = True  # tag the frames so the stub can tell them apart

    def stub_bands(frame, *args):
        return bands[0 if frame[0, 0] else 1]

    monkeypatch.setattr(seeding, "geodesic_bands", stub_bands)
    loose = choose_seed(frames, core, PIVOT, SCALE, truck_box=truck)
    strict = choose_seed(frames, core, PIVOT, SCALE, truck_box=truck, clear_of_truck=True)
    assert loose is not None and loose.sample == 0, "the split band must be the best scorer"
    assert strict is not None and strict.sample == 1


def test_without_a_span_the_first_seed_is_ranked_exactly_as_before():
    """The new arguments default to the old behaviour, sample for sample."""
    core = machine((200, 150))[1]
    masks = [machine((190 + 8 * i, 150))[0] for i in range(7)]
    old = choose_seed(masks, core, PIVOT, SCALE)
    new = choose_seed(masks, core, PIVOT, SCALE, positions=range(7))
    assert old is not None and new is not None
    assert (old.sample, old.score) == (new.sample, new.score)


# --- a limited SAM run -----------------------------------------------------------


def test_a_limited_run_tracks_the_span_and_nothing_outside_it():
    new_session = session_factory()
    masks, confidences = track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(100),
        BUCKET_OBJECT_ID,
        60,
        {"input_masks": [blob()]},
        lambda output: {BUCKET_OBJECT_ID: (blob(), 0.9)},
        keep_empty=False,
        limit=(50, 72),
    )
    forward, backward = new_session.made
    assert forward.fed == list(range(60, 73)), "forward to the span's last sample"
    assert backward.fed == list(range(60, 49, -1)), "backward to the span's first"
    assert sorted(masks) == list(range(50, 73))
    assert sorted(confidences) == list(range(50, 73))


def test_a_reseed_on_the_spans_first_sample_does_not_run_backward():
    new_session = session_factory()
    track_object(
        stub_model(),
        stub_processor(),
        new_session,
        fake_frames(100),
        BUCKET_OBJECT_ID,
        50,
        {"input_masks": [blob()]},
        lambda output: {BUCKET_OBJECT_ID: (blob(), 0.9)},
        limit=(50, 72),
    )
    assert len(new_session.made) == 1
    assert new_session.made[0].fed == list(range(50, 73))
