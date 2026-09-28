"""Tests for ``eval/check_cues.py``, the cue checker.

The checker is evaluation scaffolding: it reads the hand labels, so it must stay
outside the pipeline and must not import it. These tests pin that separation, the
shape reading on signals whose answer is known by construction, the first-match
scan, and the results on the 83 s clip that the cue work so far relies on.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "eval" / "check_cues.py"
LONG_LABELS = REPO / "eval" / "labels_long_clip.json"
DEV_LABELS = REPO / "eval" / "labels.json"
DEV_FEATURES = REPO / "tests" / "fixtures" / "dev_clip" / "features.npz"


@pytest.fixture(scope="module")
def cc():
    """Import eval/check_cues.py by path -- eval/ is not on the import path by design."""
    spec = importlib.util.spec_from_file_location("check_cues", CHECK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ separation


def test_checker_imports_nothing_from_the_pipeline():
    tree = ast.parse(CHECK.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(n.startswith("excavator_cycles") for n in names), names


# ---------------------------------------------------------------------- labels


def test_long_clip_labels_cycle_in_order(cc):
    onsets = cc.load_onsets(LONG_LABELS)
    assert len(onsets) == 13
    assert [p for p, _ in onsets] == list(cc.PHASES) * 3 + ["digging"]
    assert onsets[0][1] == pytest.approx(16 / 30)
    assert onsets[-1][1] == pytest.approx(2360 / 30)
    assert json.loads(LONG_LABELS.read_text())["cycle_count"] == 3


def test_dev_clip_labels_read_through_the_old_layout(cc):
    onsets = cc.load_onsets(DEV_LABELS)
    fps = 29.97396912419384
    assert [p for p, _ in onsets] == [*cc.PHASES, "digging"]
    assert [round(s * fps) for _, s in onsets] == [125, 322, 559, 691, 880]


# ------------------------------------------------------------------ derivative


def test_derivative_matches_the_pipeline(cc):
    """The checker re-implements the pipeline's derivative so it need not import
    it. The test may import the pipeline; this is where the two are held equal."""
    from excavator_cycles.onsets import derivative

    z = np.load(DEV_FEATURES)
    t = z["time_seconds"]
    for column in ("dx_dt", "bucket_y", "dh_dt"):
        ours = cc.derivative(z[column], t)
        theirs = derivative(z[column], t, window_seconds=0.9)
        ok = np.isfinite(theirs)
        np.testing.assert_allclose(ours[ok], theirs[ok], atol=1e-12)


# ----------------------------------------------------------------------- shape


def _ramp(t, knots):
    """Piecewise-linear signal through (time, value) knots."""
    ks, vs = zip(*knots, strict=True)
    return np.interp(t, ks, vs)


@pytest.mark.parametrize(
    ("knots", "expected"),
    [
        ([(0, 1), (10, 0), (20, 0)], "drop ends -> flat"),
        ([(0, 0), (10, 1), (20, 1)], "rise ends -> flat"),
        ([(0, 0), (10, 0), (20, 1)], "flat -> starts rising"),
        ([(0, 1), (10, 1), (20, 0)], "flat -> starts falling"),
        ([(0, 0), (10, 1), (20, 0)], "peak"),
        ([(0, 1), (10, 0), (20, 1)], "dip"),
        ([(0, 0), (20, 1)], "keeps rising"),
        ([(0, 1), (20, 0)], "keeps falling"),
    ],
)
def test_shape_at_a_known_corner(cc, knots, expected):
    t = np.arange(0, 20.01, 0.2)
    read = cc.shapes(_ramp(t, knots), t)
    assert read[int(np.argmin(abs(t - 10)))][0] == expected


def test_speeds_up_and_slows_down(cc):
    """Slopes are fractions of the range over SIDE seconds; 1.5x is the cut."""
    assert cc.shape_at(0.2, 0.5) == ("keeps rising", "speeds up")
    assert cc.shape_at(0.5, 0.2) == ("keeps rising", "slows down")
    assert cc.shape_at(-0.2, -0.5) == ("keeps falling", "speeds up")
    assert cc.shape_at(0.2, 0.25) == ("keeps rising", "")


def test_edges_have_no_shape(cc):
    t = np.arange(0, 20.01, 0.2)
    read = cc.shapes(_ramp(t, [(0, 0), (20, 1)]), t)
    assert read[0] == (None, "") and read[-1] == (None, "")


# ----------------------------------------------------------------- first match


def test_first_match_ignores_shapes_before_the_previous_phase(cc):
    """A shape that occurs before the scan starts must not count; one after the
    scan starts but before the truth is reported as early."""
    t = np.arange(150) * 0.2
    hit = np.zeros(len(t), bool)
    hit[20:26] = True  # 4.0-5.0 s, before the previous phase: ignored
    hit[99:103] = True  # 19.8-20.4 s, the real one
    labelled = [("digging", 10.0), ("hauling", 20.0)]
    (m,) = cc.first_matches(t, labelled, "hauling", hit, tolerance=0.6)
    assert m.contains and m.stretch == pytest.approx((19.8, 20.4))

    hit[60:66] = True  # 12.0-13.0 s, after the previous phase: fires early
    (m,) = cc.first_matches(t, labelled, "hauling", hit, tolerance=0.6)
    assert not m.contains and m.start_error == pytest.approx(-8.0)


def test_no_match_is_a_miss_not_a_crash(cc):
    t = np.arange(0, 30, 0.2)
    labelled = [("digging", 10.0), ("hauling", 20.0)]
    (m,) = cc.first_matches(t, labelled, "hauling", np.zeros(len(t), bool), tolerance=0.6)
    assert m.stretch is None and not m.contains and m.start_error is None


# ------------------------------------------- results the cue work relies on


@pytest.mark.parametrize(
    ("phase", "feature", "shape", "start_errors"),
    [
        ("digging", "speed_2d", "drop ends -> flat", [-0.3, -0.6, 0.0]),
        ("swinging", "d2x_dt2", "peak", [-0.2, -0.2, -0.3]),
        # ("swinging", "truck_overlap", "drop ends -> flat") is gone: the swing no
        # longer uses an overlap cue (Aanya, 2026-09-28: "this isn't a requirement
        # for swinging"), and truck_overlap now means "over the truck", not box
        # overlap in the picture.
    ],
)
def test_long_clip_cues_found_so_far(cc, phase, feature, shape, start_errors):
    t, feats = cc.load_features(cc.CLIPS["long"][0])
    labelled = cc.load_onsets(LONG_LABELS)
    read = cc.shapes(feats[feature], t)
    ms = cc.first_matches(t, labelled, phase, cc.hits(read, shape), tolerance=0.6)
    assert all(m.contains for m in ms)
    assert [round(m.start_error, 1) for m in ms] == pytest.approx(start_errors, abs=0.05)


def test_the_return_swing_cue_needs_the_dump_start(cc):
    """Scanned from the haul start instead, the d2x/dt2 peak fires far too early --
    the reason a dump cue is the blocker."""
    t, feats = cc.load_features(cc.CLIPS["long"][0])
    labelled = cc.load_onsets(LONG_LABELS)
    read = cc.shapes(feats["d2x_dt2"], t)
    ms = cc.first_matches(t, labelled, "swinging", cc.hits(read, "peak"), 0.6, anchor_back=2)
    assert not any(m.contains for m in ms)


# ------------------------------------------------------------------------- CLI


def test_cli_exit_codes(cc, capsys):
    assert (
        cc.main(
            ["--phase", "digging", "--feature", "speed_2d", "--shape", "drop ends -> flat"]
        )
        == 0
    )
    assert (
        cc.main(
            [
                "--phase",
                "swinging",
                "--feature",
                "d2x_dt2",
                "--shape",
                "peak",
                "--anchor-back",
                "2",
            ]
        )
        == 1
    )
    assert cc.main(["--phase", "nope"]) == 2
    assert cc.main(["--phase", "digging", "--feature", "nope", "--shape", "peak"]) == 2
    assert "unknown feature" in capsys.readouterr().err


def test_cli_accepts_the_arrow_and_short_names(cc):
    assert (
        cc.main(["--phase", "dig", "--feature", "speed_2d", "--shape", "drop ends → flat"])
        == 0
    )


def test_cli_search_runs_on_the_dev_clip(cc, capsys):
    cc.main(["--clip", "dev", "--phase", "swinging", "--search", "--top", "3"])
    out = capsys.readouterr().out
    assert "swinging: every feature x shape" in out and "dev_clip" in out


def test_the_checker_reads_shapes_as_the_pipeline_does(cc):
    """Two implementations of one reading -- the pipeline may not import eval, and
    the checker does not import the pipeline -- held equal wherever both read a
    shape. They differ only at the clip edges, where the pipeline fits a shorter
    side and the checker reports nothing."""
    from excavator_cycles.shapes import read

    t, feats = cc.load_features(cc.CLIPS["long"][0])
    for name in ("height", "speed_2d", "d2x_dt2", "bucket_x"):
        ours = cc.shapes(feats[name], t)
        theirs = read(feats[name], t, side=2.0, flat=0.08, steeper=1.5, min_side=0.4)
        both = [i for i, (s, _) in enumerate(ours) if s is not None]
        assert len(both) > 0.9 * len(t)
        assert [ours[i] for i in both] == [
            (theirs.shape[i], theirs.detail[i]) for i in both
        ], name
