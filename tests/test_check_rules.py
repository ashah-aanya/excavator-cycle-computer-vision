"""Tests for the hard and soft rules in ``eval/check_cues.py``.

Uses the 29.6 s dev clip only. The rules are Aanya's classification
(docs/stages/09-rules-draft.md): hard rules veto a candidate moment, soft rules are
evidence only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location(
        "check_cues", REPO / "eval" / "check_cues.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dev(cc):
    features, labels = cc.CLIPS["dev"]
    t, feats = cc.load_features(features)
    return t, feats, cc.load_onsets(labels)


def test_every_phase_has_its_rules_declared(cc):
    assert set(cc.RULES) == set(cc.PHASES)
    assert cc.RULES["hauling"]["hard"] == ("H1",)
    assert cc.RULES["dumping"]["hard"] == ("P1", "P2")
    assert cc.RULES["digging"]["hard"] == ("D1", "D2")


def test_the_truck_is_on_the_plus_side_of_the_dev_clip(cc, dev):
    _, feats, _ = dev
    assert cc.truck_side(feats) == 1.0


def test_hard_rules_never_veto_a_labelled_onset(cc, dev):
    """A veto that blocks the right answer is worse than no veto. Every hard rule
    must hold for the first 0.6 s of its phase at every labelled onset."""
    t, feats, labelled = dev
    side = cc.truck_side(feats)
    for k, (phase, start) in enumerate(labelled):
        masks = cc.rule_masks(t, feats, labelled, k, side)
        window = (t >= start) & (t < start + 0.6)
        for rid in cc.RULES[phase]["hard"]:
            assert masks[rid][window].all(), f"{rid} vetoes the {phase} onset at {start:.2f} s"


def test_the_dig_memory_uses_only_the_past(cc):
    values = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    med = cc._running_median(values, 0)
    assert np.isnan(med[0])
    assert med[1] == 0.0 and med[2] == 0.5 and med[3] == 1.0
    assert med[4] == 1.5, "the 100 at the current sample must not be included"


def test_a_hard_rule_vetoes_a_match_it_rules_out(cc, dev):
    """A cue firing in the dig, where the bucket is no higher than the dig itself,
    is refused by H1; the real haul onset is kept."""
    t, feats, labelled = dev
    haul_start = next(s for p, s in labelled if p == "hauling")
    dig_start = next(s for p, s in labelled if p == "digging")
    hit = np.zeros(len(t), bool)
    early = (t >= dig_start + 1.0) & (t < dig_start + 1.4)
    assert (feats["height"][early] < np.median(feats["height"][t < haul_start])).all()
    hit[early] = True  # a false match early in the dig, bucket still low
    hit[(t >= haul_start) & (t < haul_start + 0.4)] = True  # the real one
    plain = cc.first_matches(t, labelled, "hauling", hit, 0.6)
    ruled = cc.first_matches_ruled(t, feats, labelled, "hauling", hit, 0.6)
    assert not plain[0].contains, "without the rule the early match wins"
    assert ruled[0][0].contains, "with H1 the early match is vetoed"


def test_soft_rules_are_reported_not_applied(cc, dev):
    t, feats, labelled = dev
    hit = cc.hits(cc.shapes(feats["height"], t), "keeps rising")
    (_match, soft) = cc.first_matches_ruled(t, feats, labelled, "hauling", hit, 0.6)[0]
    assert set(soft) == {"H3"}


# --- big drops: the dump's tipping onset (dev clip only) ----------------------


def test_a_big_drop_is_found_where_it_starts(cc):
    """A plateau that falls by its whole spread in 1 s: the onset is where the fall
    leaves the plateau, the bottom where it lands."""
    t = np.arange(0, 20.01, 0.1)
    v = np.interp(t, [0, 10, 11, 20], [2.0, 2.0, 1.0, 1.0])
    (d,) = cc.big_drops(v, t)
    assert d["onset"] == pytest.approx(10.0, abs=0.15)
    assert d["bottom"] == pytest.approx(11.0, abs=0.15)
    assert d["drop"] > 0.9


def test_a_small_or_slow_fall_is_not_a_big_drop(cc):
    t = np.arange(0, 20.01, 0.1)
    small = np.interp(t, [0, 10, 11, 20], [2.0, 2.0, 1.8, 1.8]) + np.interp(
        t, [0, 5, 20], [0, 1, 1]
    )  # the 0.2 fall is a fifth of a spread widened by a 1.0 rise
    slow = np.interp(t, [0, 5, 15, 20], [2.0, 2.0, 1.0, 1.0])  # takes 10 s
    assert cc.big_drops(small, t) == []
    assert cc.big_drops(slow, t) == []


def test_the_dev_clip_tips_once_after_the_haul(cc, dev):
    """Aanya's definition (2026-09-27): the dump starts when the bucket starts to
    tip, read from the start of the aspect ratio's big drop. On the dev clip the
    frames show the bucket curled at 20.3 s and visibly tipping at 20.6 s."""
    t, feats, labelled = dev
    haul = next(s for p, s in labelled if p == "hauling")
    after = [d for d in cc.big_drops(feats["aspect_ratio"], t) if d["onset"] >= haul]
    assert len(after) == 1, "one tip per cycle, and nothing else that looks like one"
    assert after[0]["onset"] == pytest.approx(20.62, abs=0.05)


# --- window finders: knee and end of a big dip (dev clip only) ------------------


def test_a_take_off_knee_is_found_at_the_corner(cc):
    t = np.arange(0, 20.01, 0.1)
    v = np.interp(t, [0, 10, 14, 20], [0.0, 0.0, 1.0, 1.0])
    (w,) = cc.knee_windows(v, t, "rising", "flat_to_steep")
    assert w[0] <= 10.0 <= w[1] + 0.2


def test_a_landing_knee_is_found_at_the_corner(cc):
    t = np.arange(0, 20.01, 0.1)
    v = np.interp(t, [0, 6, 10, 20], [1.0, 1.0, 0.0, 0.0])
    (w,) = cc.knee_windows(v, t, "falling", "steep_to_flat")
    assert w[0] - 0.2 <= 10.0 <= w[1]


def test_a_straight_line_has_no_knee(cc):
    t = np.arange(0, 20.01, 0.1)
    assert cc.knee_windows(t * 0.05, t, "rising", "flat_to_steep") == []


def test_the_end_of_a_big_dip_spans_bottom_to_recovery(cc):
    t = np.arange(0, 20.01, 0.1)
    v = np.interp(t, [0, 8, 10, 12, 20], [0.0, 0.0, -1.0, 0.0, 0.0])
    (d,) = cc.excursion_windows(v, t, "dip")
    assert d["apex"] == pytest.approx(10.0, abs=0.15)
    assert d["start"] == pytest.approx(8.0, abs=0.15)
    assert d["end"] == pytest.approx(12.0, abs=0.15)


def test_a_small_dip_is_not_a_big_one(cc):
    t = np.arange(0, 20.01, 0.1)
    v = np.interp(t, [0, 4, 5, 6, 12, 14, 16, 20], [0, 0, -0.2, 0, 0, -1.0, 0, 0])
    assert len(cc.excursion_windows(v, t, "dip")) == 1


def test_the_dev_clip_has_one_big_dx_dip_per_dig(cc, dev):
    """dx/dt dips hard as the bucket swings back and brakes onto the pile. On the
    dev clip there are exactly two such dips, and each end-of-dip window (bottom
    to recovery) contains a labelled dig start."""
    t, feats, labelled = dev
    digs = [s for p, s in labelled if p == "digging"]
    wins = [(d["apex"], d["end"]) for d in cc.excursion_windows(feats["dx_dt"], t, "dip")]
    assert len(wins) == 2
    for (a, b), dig in zip(wins, digs, strict=True):
        assert a <= dig <= b
