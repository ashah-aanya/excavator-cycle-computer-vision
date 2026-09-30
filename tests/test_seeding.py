"""Tests for geodesic bucket seeding (design doc section 3, bucket mask).

The claim under test is narrow and worth stating plainly: on a frame *we get to
choose*, the far end of the arm measured **along the metal** is the bucket, and
points placed in its interior are a safe thing to hand SAM 2.

The interesting cases are the ones that separate that claim from the failure it
resembles -- ``farthest_point`` per frame, which was the first of four failed
attempts at arm pose:

* a raised boom must NOT outrank the bucket (this is the one Euclidean distance
  gets wrong, by 70 px on the development video);
* points must land INSIDE the bucket, not on its rim, because mask boundaries
  are the least reliable part of a mask;
* a frame beside the truck must lose to one clear of it;
* the last frame of a clip must not win on reach alone, or SAM ends up
  reverse-propagating the whole video.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.config import Config
from excavator_cycles.seeding import (
    choose_seed,
    extension,
    geodesic_bands,
    interior_points,
    looks_like_the_boom_apex,
    reach_score,
    score_frame,
)

SHAPE = (240, 320)
PIVOT = (60.0, 180.0)
SCALE = 120.0


def machine(
    elbow: tuple[int, int] = (150, 90),
    bucket: tuple[int, int] = (240, 150),
    bucket_radius: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    """A body blob at the pivot, a two-segment arm, and a blob for the bucket.

    Deliberately shaped like the real thing: the body is compact and sits ON the
    pivot, the arm reaches well past it, and the bucket is a blob at the far
    end. A body as long as the arm would not be separable by distance at all,
    and a test that pretended otherwise would be lying.
    """
    mask = np.zeros(SHAPE, bool)
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]

    body = np.hypot(xs - PIVOT[0], ys - PIVOT[1]) <= 34
    mask |= body

    for start, end in ((PIVOT, elbow), (elbow, bucket)):
        steps = int(np.hypot(end[0] - start[0], end[1] - start[1])) * 2
        for t in np.linspace(0, 1, max(steps, 2)):
            cx = start[0] + t * (end[0] - start[0])
            cy = start[1] + t * (end[1] - start[1])
            mask |= np.hypot(xs - cx, ys - cy) <= 7

    mask |= np.hypot(xs - bucket[0], ys - bucket[1]) <= bucket_radius
    return mask, body


# --- the bands -------------------------------------------------------------


def test_the_bucket_band_lands_on_the_bucket():
    """The far end along the metal is the bucket, not some slice of arm."""
    bucket_at = (240, 150)
    mask, core = machine(bucket=bucket_at)
    band, stick, reach = geodesic_bands(mask, core, PIVOT)

    assert band.any(), "no bucket band was produced"
    ys, xs = np.nonzero(band)
    centre = (xs.mean(), ys.mean())
    offset = np.hypot(centre[0] - bucket_at[0], centre[1] - bucket_at[1])
    assert offset < 25, f"band centre {centre} is {offset:.0f}px from the bucket"
    assert reach > 0
    assert stick.any(), "no stick band was produced"


def test_a_raised_boom_does_not_outrank_the_bucket():
    """The failure Euclidean distance has and geodesic distance cannot.

    With the elbow raised high, the boom's apex is FARTHER from the pivot in a
    straight line than the bucket is -- measured at 70px on the development
    video. Along the metal it can never be, because the apex is a point part
    way down a path that continues to the bucket.
    """
    bucket_at = (150, 200)
    mask, core = machine(elbow=(120, 20), bucket=bucket_at)

    apex = np.array([120.0, 20.0])
    euclid_apex = np.hypot(apex[0] - PIVOT[0], apex[1] - PIVOT[1])
    euclid_bucket = np.hypot(bucket_at[0] - PIVOT[0], bucket_at[1] - PIVOT[1])
    assert euclid_apex > euclid_bucket, "fixture does not reproduce the trap"

    band, _, _ = geodesic_bands(mask, core, PIVOT)
    ys, xs = np.nonzero(band)
    centre = (xs.mean(), ys.mean())
    to_bucket = np.hypot(centre[0] - bucket_at[0], centre[1] - bucket_at[1])
    to_apex = np.hypot(centre[0] - apex[0], centre[1] - apex[1])
    assert to_bucket < to_apex, (
        f"band centre {centre} sits nearer the boom apex ({to_apex:.0f}px) "
        f"than the bucket ({to_bucket:.0f}px)"
    )


def test_an_empty_mask_yields_empty_bands():
    empty = np.zeros(SHAPE, bool)
    band, stick, reach = geodesic_bands(empty, empty, PIVOT)
    assert not band.any() and not stick.any() and reach == 0.0


# --- the points ------------------------------------------------------------


def test_points_land_inside_the_region_not_on_its_rim():
    """Boundaries deform by 10-30px during overlap; interiors do not."""
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    disc = np.hypot(xs - 160, ys - 120) <= 30

    points = interior_points(disc, 5)
    assert len(points) == 5
    for x, y in points:
        assert disc[round(y), round(x)], f"({x},{y}) is outside the region"
        assert np.hypot(x - 160, y - 120) < 24, "point sits on the rim"


def test_points_are_spread_rather_than_clumped():
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    disc = np.hypot(xs - 160, ys - 120) <= 30

    points = interior_points(disc, 4)
    gaps = [
        np.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(points)
        for b in points[i + 1 :]
    ]
    assert min(gaps) > 6, f"closest pair is only {min(gaps):.1f}px apart"


def test_a_region_too_small_to_trust_yields_no_points():
    tiny = np.zeros(SHAPE, bool)
    tiny[100:103, 100:103] = True
    assert len(interior_points(tiny, 5)) == 0


# --- choosing the frame ----------------------------------------------------


def test_a_frame_beside_the_truck_loses_to_one_clear_of_it():
    """Seeding next to the truck invites SAM to claim truck pixels."""
    near, core = machine(bucket=(250, 150))
    far, _ = machine(bucket=(150, 200))
    truck = (230.0, 100.0, 310.0, 200.0)

    near_band, _, near_reach = geodesic_bands(near, core, PIVOT)
    far_band, _, far_reach = geodesic_bands(far, core, PIVOT)

    near_score = score_frame(near_band, near_reach, SCALE, truck, 5, 10)
    far_score = score_frame(far_band, far_reach, SCALE, truck, 5, 10)
    assert far_score > near_score, f"near={near_score:.3f} far={far_score:.3f}"


def test_the_last_frame_does_not_win_on_reach_alone():
    """The naive maximum-reach rule picked sample 295 of 296 on the real clip.

    SAM 2 propagates both ways from its seed, so seeding at the very end turns
    the whole video into one long reverse pass.
    """
    mask, core = machine()
    band, _, reach = geodesic_bands(mask, core, PIVOT)

    middle = score_frame(band, reach, SCALE, None, 50, 100)
    last = score_frame(band, reach, SCALE, None, 99, 100)
    assert middle > last, f"middle={middle:.3f} last={last:.3f}"


def test_choose_seed_returns_the_best_frame_with_every_prompt_form():
    """SAM 2 prices prompts very differently, so all three forms are returned."""
    core = machine()[1]
    masks = [machine(bucket=(200 + 10 * i, 150))[0] for i in range(6)]

    seed = choose_seed(masks, core, PIVOT, SCALE, point_count=4, negative_count=2)
    assert seed is not None
    assert 0 <= seed.sample < len(masks)
    assert len(seed.points) == 4
    assert seed.band.any()
    x0, y0, x1, y1 = seed.box
    assert x1 > x0 and y1 > y0
    for x, y in seed.points:
        assert x0 <= x < x1 and y0 <= y < y1, "a point escaped its own box"


def test_negatives_sit_on_the_arm_not_the_bucket():
    """Negatives are what stop SAM claiming the whole arm as 'the bucket'."""
    core = machine()[1]
    masks = [machine()[0]]

    seed = choose_seed(masks, core, PIVOT, SCALE, negative_count=3)
    assert seed is not None
    assert len(seed.negatives) > 0
    for x, y in seed.negatives:
        assert not seed.band[round(y), round(x)], (
            f"negative ({x},{y}) landed inside the bucket band"
        )


def test_no_usable_frame_returns_none_rather_than_guessing():
    empty = np.zeros(SHAPE, bool)
    assert choose_seed([empty, empty], empty, PIVOT, SCALE) is None


# --- the one failure geodesic distance genuinely has ------------------------


def test_a_band_on_the_boom_apex_is_rejected_outright():
    """When the arm folds so the bucket touches the cab, the path short-circuits.

    The shape really is ambiguous there -- two parts of the machine are in
    contact in projection -- so no band edge fixes it. What can be done is
    refusing to SEED from such a frame. On an excavator the apex is the top of
    the silhouette by construction and the bucket hangs below it, so a band
    whose centroid sits at the machine's highest row is the apex.

    Scoring alone was not enough: the worst such frame still ranked #110 of 287
    on the development video, because reach demotes but does not exclude it.
    """
    mask, core = machine()
    band, _, reach = geodesic_bands(mask, core, PIVOT)

    top_row = np.nonzero(mask)[0].min()
    apex_band = np.zeros_like(mask)
    apex_band[top_row : top_row + 10, 140:165] = True

    assert looks_like_the_boom_apex(apex_band, mask, SCALE)
    assert not looks_like_the_boom_apex(band, mask, SCALE), "the real bucket band was flagged"
    assert score_frame(apex_band, reach, SCALE, None, 5, 10, mask) == 0.0
    assert score_frame(band, reach, SCALE, None, 5, 10, mask) > 0.0


def test_a_raised_bucket_is_not_mistaken_for_the_apex():
    """Dumping raises the bucket high, and that must still be a usable frame.

    Measured on the development video: this check fires on 19% of hauling and
    on 0 of 44 dumping samples, because the boom stays above the bucket even
    when the bucket is up.
    """
    mask, core = machine(elbow=(150, 40), bucket=(230, 80))
    band, _, _ = geodesic_bands(mask, core, PIVOT)

    assert band.any()
    assert not looks_like_the_boom_apex(band, mask, SCALE), (
        "a legitimately raised bucket was rejected as the boom apex"
    )


# --- the "reach" frame rule --------------------------------------------------


def stretched_and_folded():
    """Six frames, the arm stretched furthest at index 0 -- the edge of the clip."""
    core = machine()[1]
    buckets = [(285, 150), (240, 150), (215, 150), (195, 150), (180, 150), (170, 150)]
    return core, [machine(bucket=b)[0] for b in buckets]


def test_the_reach_rule_picks_the_most_stretched_arm():
    core, masks = stretched_and_folded()
    reaches = [extension(m, PIVOT) for m in masks]
    assert reaches[0] == max(reaches), "fixture: frame 0 should be the most stretched"

    seed = choose_seed(masks, core, PIVOT, SCALE, rule="reach")
    assert seed is not None and seed.sample == 0


def test_the_reach_rule_does_not_prefer_the_middle_of_the_clip():
    """With arms of nearly equal length the older score follows its mid-clip bonus.

    ``score_frame`` adds up to 0.5 for a frame in the middle of the clip, which outweighs a
    few pixels of extra reach; "reach" is the arm alone, as tested offline. Frame 0 is the
    edge of the clip and only slightly more stretched than the rest.
    """
    core = machine()[1]
    buckets = [(245, 150), (240, 150), (240, 150), (240, 150), (240, 150), (240, 150)]
    masks = [machine(bucket=b)[0] for b in buckets]
    assert extension(masks[0], PIVOT) > max(extension(m, PIVOT) for m in masks[1:])

    reach_pick = choose_seed(masks, core, PIVOT, SCALE, rule="reach")
    score_pick = choose_seed(masks, core, PIVOT, SCALE, rule="score")
    assert reach_pick is not None and score_pick is not None
    assert reach_pick.sample == 0
    assert score_pick.sample != 0


def test_the_default_rule_is_still_the_older_score():
    core, masks = stretched_and_folded()
    default = choose_seed(masks, core, PIVOT, SCALE)
    explicit = choose_seed(masks, core, PIVOT, SCALE, rule="score")
    assert default is not None and explicit is not None
    assert default.sample == explicit.sample and default.score == explicit.score


def test_the_reach_score_is_a_fraction_of_the_arms_reach_and_ranks_by_extension():
    mask, core = machine()
    band = geodesic_bands(mask, core, PIVOT)[0]
    assert reach_score(band, mask, PIVOT, SCALE) == pytest.approx(
        extension(mask, PIVOT) / SCALE
    )
    longer = machine(bucket=(285, 150))[0]
    longer_band = geodesic_bands(longer, core, PIVOT)[0]
    assert reach_score(longer_band, longer, PIVOT, SCALE) > reach_score(
        band, mask, PIVOT, SCALE
    )


def test_a_band_on_the_boom_apex_still_scores_zero_under_the_reach_rule():
    """A raised boom must not win just because it reaches far in a straight line."""
    mask, core = machine()
    top_row = np.nonzero(mask)[0].min()
    apex_band = np.zeros_like(mask)
    apex_band[top_row : top_row + 10, 140:165] = True
    real_band = geodesic_bands(mask, core, PIVOT)[0]
    assert reach_score(apex_band, mask, PIVOT, SCALE) == 0.0
    assert reach_score(real_band, mask, PIVOT, SCALE) > 0.0


def test_an_empty_mask_or_band_has_no_reach_score():
    mask, core = machine()
    band = geodesic_bands(mask, core, PIVOT)[0]
    empty = np.zeros(SHAPE, bool)
    assert reach_score(empty, mask, PIVOT, SCALE) == 0.0
    assert reach_score(band, empty, PIVOT, SCALE) == 0.0
    assert extension(empty, PIVOT) == 0.0


def test_an_unknown_frame_rule_is_refused():
    core, masks = stretched_and_folded()
    with pytest.raises(ValueError, match="frame rule"):
        choose_seed(masks, core, PIVOT, SCALE, rule="sharpest")


def test_the_configured_frame_rule_reaches_the_seed_choice(monkeypatch):
    """track.bucket_frame_rule must be what derive_bucket_seed hands to choose_seed."""
    from excavator_cycles import track

    seen = []
    monkeypatch.setattr(
        track.seeding, "choose_seed", lambda *args, **kwargs: seen.append(kwargs["rule"])
    )
    masks = {i: machine(bucket=(240 + i, 150))[0] for i in range(4)}

    assert Config().track.bucket_frame_rule == "reach"
    for rule in ("reach", "score"):
        config = Config.load(overrides={"track": {"bucket_frame_rule": rule}})
        track.derive_bucket_seed(masks, len(masks), None, config)
    assert seen == ["reach", "score"]
