"""The phase search when the clip ends with no dig left to find.

A cycle runs dig -> haul -> dump -> swing and then starts again at a dig. A clip that stops
filming after a clean swing has only a short tail left, too short to hold a dig. The search
must treat "no dig after here" as the end of the footage and report it, the way it already
does when a haul, dump or swing is not found. It used to read ``s["window"]`` on ``None``
and raise a TypeError.

These tests drive ``from_found`` with stubbed stage results, so they state the rule without
a video: the crash depends on what the stages return, not on any pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles import windows

# One clean cycle, then 36 s of clip with no further dig.
DIG, HAUL, DUMP, SWING = (2.0, 4.0), (5.0, 7.0), (8.0, 10.0), (11.0, 13.0)
CYCLE = {"hauling": HAUL, "dumping": DUMP, "swinging": SWING}


def _stage(phase, window):
    return {
        "phase": phase,
        "window": window,
        "core": window,
        "weak": False,
        "votes": 3,
        "need": 2.5,
    }


@pytest.fixture
def one_cycle_then_nothing(monkeypatch):
    """The stages find one full cycle; every later search for a dig comes back empty."""

    def find_dig(t, F, found, anchor, cycle, reach=2, weak=True):
        return _stage("digging", DIG) if anchor < DIG[0] else None

    def step(t, F, found, phase, anchor, cycle):
        return _stage(phase, CYCLE[phase])

    monkeypatch.setattr(windows, "finders", lambda t, F, side: {})
    monkeypatch.setattr(windows, "find_dig", find_dig)
    monkeypatch.setattr(windows, "step", step)


def test_a_clip_that_ends_after_a_swing_stops_instead_of_crashing(one_cycle_then_nothing):
    t = np.arange(0.0, 50.0, 0.1)

    steps, stop, gaps = windows.from_found(t, {}, 1.0)

    assert [s["phase"] for s in steps] == ["digging", "hauling", "dumping", "swinging"]
    assert stop is not None
    assert stop["phase"] == "digging"
    assert stop["search_from"] == pytest.approx(SWING[1])  # looked from where the swing ended
    assert "no dig" in stop["why"]
    assert gaps == []


def test_a_clip_with_no_dig_at_all_reports_it(monkeypatch):
    monkeypatch.setattr(windows, "finders", lambda t, F, side: {})
    monkeypatch.setattr(windows, "find_dig", lambda *a, **k: None)
    t = np.arange(0.0, 50.0, 0.1)

    steps, stop, _gaps = windows.from_found(t, {}, 1.0)

    assert steps == []
    assert stop is not None
    assert stop["phase"] == "digging"
    assert stop["search_from"] == pytest.approx(0.0)
    assert "no dig" in stop["why"]
