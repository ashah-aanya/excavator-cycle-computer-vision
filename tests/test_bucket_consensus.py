"""Tests for ``eval/bucket_consensus.py``, the bucket-seed consensus experiment.

Every case is built from a synthetic arm whose right answer is known by construction: a
body block, a straight stick of fixed thickness, and (or not) a thick bucket at the end.
The tests pin what each estimator returns, how agreement is scored, and that a frame is only
trusted when at least two independent estimators support the same region.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from excavator_cycles.kinematics import boom_base

MODULE = Path(__file__).resolve().parents[1] / "eval" / "bucket_consensus.py"
spec = importlib.util.spec_from_file_location("bucket_consensus", MODULE)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

HEIGHT, WIDTH = 200, 320


def arm(with_bucket: bool = True):
    """Body block at the left, a 12 px stick to the right, and a 34 px bucket at its end."""
    core = np.zeros((HEIGHT, WIDTH), bool)
    core[110:150, 20:60] = True
    mask = core.copy()
    mask[118:130, 60:200] = True  # the stick, 12 px thick
    bucket = np.zeros_like(mask)
    if with_bucket:
        bucket[104:138, 165:205] = True
        mask |= bucket
    return mask, core, bucket


def test_geodesic_and_thickness_both_land_on_the_bucket_and_agree():
    mask, core, bucket = arm()
    pivot = boom_base(core)
    band = bc.estimate_geodesic(mask, core, pivot)
    thick = bc.estimate_thickness(mask, core, pivot)
    assert band is not None and thick is not None
    assert np.logical_and(band, bucket).sum() / band.sum() > 0.6  # the band is on the bucket
    assert np.logical_and(thick, bucket).sum() / bucket.sum() > 0.6  # the thick end covers it
    assert bc.agreement(band, thick) > 0.6


def test_an_arm_with_no_bucket_in_the_mask_gives_no_thickness_estimate():
    """A uniform stick has no thick end: this is the failure the consensus must notice."""
    mask, core, _ = arm(with_bucket=False)
    pivot = boom_base(core)
    assert bc.estimate_thickness(mask, core, pivot) is None
    band = bc.estimate_geodesic(mask, core, pivot)
    assert band is not None  # the band still exists: it is a sliver of stick
    score, region = bc.frame_consensus({"geodesic": band, "thickness": None})
    assert score == 0.0 and region is None


def test_a_dark_blob_at_the_tip_is_found_by_appearance():
    mask, core, bucket = arm()
    pivot = boom_base(core)
    rng = np.random.default_rng(0)
    image = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
    image = np.clip(image + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)
    image[mask & ~bucket] = (0, 190, 240)  # yellow paint
    image[bucket] = (35, 35, 35)  # dark bucket
    cv2.setRNGSeed(0)
    found = bc.estimate_appearance(image, mask, core, pivot)
    assert found is not None
    assert np.logical_and(found, bucket).sum() / bucket.sum() > 0.4


def test_agreement_is_symmetric_and_zero_for_disjoint_tiny_or_mismatched_regions():
    a = np.zeros((100, 100), bool)
    a[10:40, 10:40] = True
    b = np.zeros_like(a)
    b[70:95, 70:95] = True
    small = np.zeros_like(a)
    small[10:14, 10:14] = True
    huge = np.zeros_like(a)
    huge[0:100, 0:100] = True
    assert bc.agreement(a, a) == 1.0
    assert bc.agreement(a, b) == 0.0
    assert bc.agreement(a, small) == 0.0  # under the minimum size
    assert bc.agreement(small, huge) == 0.0
    assert bc.agreement(a, huge) == 0.0  # areas differ by more than the allowed ratio
    c = np.zeros_like(a)
    c[20:50, 20:50] = True
    assert bc.agreement(a, c) == bc.agreement(c, a)


def test_consensus_needs_two_estimators_and_takes_the_weakest_pair():
    a = np.zeros((100, 100), bool)
    a[10:40, 10:40] = True
    b = a.copy()
    c = np.zeros_like(a)
    c[30:60, 30:60] = True  # overlaps a by 10x10 of 900
    assert bc.frame_consensus({"a": a})[0] == 0.0
    assert bc.frame_consensus({"a": a, "b": None})[1] is None
    score, region = bc.frame_consensus({"a": a, "b": b})
    assert score == 1.0 and region.sum() == a.sum()
    score3, region3 = bc.frame_consensus({"a": a, "b": b, "c": c})
    assert score3 < 0.2  # the weakest pair (a, c) sets the score
    assert region3.sum() >= a.sum()  # pixels supported by at least two estimators


def test_pool_is_long_extension_with_agreement_at_or_above_the_median():
    rows = [
        (i, float(i), 0.9 if i % 2 else 0.4) for i in range(12)
    ]  # (sample, extension, agreement)
    pool = bc.choose_pool(rows)
    assert {r[0] for r in pool} <= {8, 9, 10, 11}  # top third by extension only
    assert all(r[2] >= 0.65 for r in pool)  # and not the low-agreement ones
    flat = [(i, float(i), 0.5) for i in range(9)]
    assert bc.choose_pool(flat)  # never empty


def test_the_experiment_reads_no_hand_labels():
    """It scores against a good run's own bucket track, never against eval/labels*.json."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    strings = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert not [s for s in strings if "labels" in s and s.endswith(".json")]
    assert "check_cues" not in MODULE.read_text(encoding="utf-8")


def test_a_required_estimator_that_abstains_zeroes_the_frame():
    """Two estimators can agree on the wrong thing; requiring the thickness one blocks that."""
    a = np.zeros((100, 100), bool)
    a[10:40, 10:40] = True
    b = a.copy()
    assert bc.frame_consensus({"geodesic": a, "appearance": b})[0] == 1.0
    score, region = bc.frame_consensus(
        {"geodesic": a, "thickness": None, "appearance": b}, require=("thickness",)
    )
    assert score == 0.0 and region is None
    assert (
        bc.frame_consensus({"geodesic": a, "thickness": b}, require=("thickness",))[0] == 1.0
    )
