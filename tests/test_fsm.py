"""The state machine: calibration, the walk, and pass 2.

Written before the implementation. Every test here states a property the design
document commits to, so a failure means either the code is wrong or the design
changed -- and the second is worth noticing.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavator_cycles.fsm import Split, calibrate, split_of

# --- what a split is ------------------------------------------------------


def _bimodal(n=400, gap=1.0, spread=0.05, seed=0):
    """Two clearly separated clumps -- the shape a phase signal has."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, spread, n // 2), rng.normal(gap, spread, n // 2)])


def test_a_clean_split_separates_the_two_modes():
    """The property is the PARTITION, not where in the valley the cut sits.

    When the gap between the modes is empty, every threshold inside it produces
    the identical split, so asserting a numeric position would be testing an
    arbitrary tie-break. What has to be true is that the two clumps end up on
    opposite sides.
    """
    signal = _bimodal(n=400, gap=1.0)
    split = split_of(signal)
    assert split.below == 200 and split.above == 200, "each mode should land on its own side"
    assert (signal[:200] < split.threshold).all(), "the low clump belongs below"
    assert (signal[200:] >= split.threshold).all(), "the high clump belongs above"


def test_a_clean_split_reports_high_separability():
    """Separability is Otsu's own objective, normalised: how much of the total
    variance the split explains. Near 1 means two clear modes."""
    assert split_of(_bimodal(gap=1.0)).separability > 0.9


def test_a_unimodal_signal_is_flagged_rather_than_silently_split():
    """Otsu ALWAYS returns a number. On one mode that number is meaningless, and
    the only defence is reporting that the split explains almost nothing."""
    rng = np.random.default_rng(1)
    split = split_of(rng.normal(0.0, 1.0, 400))
    assert split.separability < 0.8, "a single Gaussian should not look bimodal"
    assert not split.trustworthy


def test_separability_is_a_fraction():
    for signal in (_bimodal(), np.random.default_rng(2).normal(0, 1, 300)):
        assert 0.0 <= split_of(signal).separability <= 1.0


def test_a_wider_gap_separates_better():
    """The measure has to be monotone in the thing it claims to measure."""
    near = split_of(_bimodal(gap=0.3)).separability
    far = split_of(_bimodal(gap=3.0)).separability
    assert far > near


def test_the_threshold_scales_with_the_data():
    """The whole point: no absolute level. Shoot from twice as far, every length
    halves, and the cut has to follow."""
    signal = _bimodal(gap=1.0)
    assert split_of(signal * 0.5).threshold == pytest.approx(
        split_of(signal).threshold * 0.5, rel=0.05
    )


def test_nan_samples_are_ignored_not_counted():
    signal = _bimodal(gap=1.0)
    holed = signal.copy()
    holed[::7] = np.nan
    assert split_of(holed).threshold == pytest.approx(split_of(signal).threshold, rel=0.1)


def test_a_signal_with_nothing_in_it_raises():
    with pytest.raises(ValueError, match="no finite"):
        split_of(np.full(10, np.nan))


def test_a_constant_signal_is_not_trustworthy():
    """No variance means no split; it must not claim to have found one."""
    split = split_of(np.full(200, 0.4))
    assert not split.trustworthy


# --- calibrate over a whole table -----------------------------------------


class _Table:
    """Only the columns calibrate() reads."""

    def __init__(self, height, truck_overlap, speed_x, times=None):
        self.height = np.asarray(height, float)
        self.truck_overlap = np.asarray(truck_overlap, float)
        self.speed_x = np.asarray(speed_x, float)
        self.time_seconds = (
            np.arange(len(self.height)) * 0.1 if times is None else np.asarray(times, float)
        )


def _table(n=400):
    return _Table(
        height=_bimodal(n, gap=1.0),
        truck_overlap=_bimodal(n, gap=0.6, seed=3),
        speed_x=np.abs(_bimodal(n, gap=0.8, seed=4)),
    )


def test_calibrate_returns_a_level_for_each_gate():
    levels = calibrate(_table())
    assert isinstance(levels.low_height, Split)
    assert isinstance(levels.over_truck, Split)
    assert isinstance(levels.moving, Split)


def test_calibrate_is_levels_only_and_says_nothing_about_time():
    """The design's central rule: a global statistic may say what 'low' MEANS,
    and may not say WHEN anything happened. If calibrate ever returns an
    interval, the old bracket-first design has crept back in."""
    levels = calibrate(_table())
    for field in vars(levels):
        value = getattr(levels, field)
        assert not isinstance(value, (list, tuple)), (
            f"{field} looks like a segmentation, not a level"
        )


def test_calibrate_survives_a_video_with_no_truck():
    """truck_overlap is all-nan when no truck was ever detected. That is a
    legitimate video, not an error -- the gate is simply unavailable."""
    table = _table()
    table.truck_overlap = np.full(len(table.height), np.nan)
    levels = calibrate(table)
    assert levels.over_truck is None or not levels.over_truck.trustworthy


def test_separability_tells_one_population_from_two():
    """The evidence MIN_SEPARABILITY is set from, pinned so it cannot drift.

    Otsu always returns a number, so the only question that matters is whether
    the data had two populations to separate. These are the measurements the
    threshold was chosen between.
    """
    from excavator_cycles.fsm import MIN_SEPARABILITY

    rng = np.random.default_rng(0)
    one_population = {
        "gaussian": rng.normal(0, 1, 400),
        "uniform": rng.uniform(0, 1, 400),
        "exponential": rng.exponential(1, 400),
    }
    two_populations = {
        "well separated": np.r_[rng.normal(0, 0.05, 200), rng.normal(1.0, 0.05, 200)],
        # not bimodal in the Gaussian sense -- near-zero with an occasional
        # excursion, which is the shape truck_overlap and |dx/dt| actually have
        "zero inflated": np.r_[np.zeros(220), rng.uniform(0.05, 0.4, 80)],
    }
    for name, signal in one_population.items():
        assert split_of(signal).separability < MIN_SEPARABILITY, f"{name} should look weak"
    for name, signal in two_populations.items():
        assert split_of(signal).separability >= MIN_SEPARABILITY, f"{name} should look real"


# --- the thing that remembers ---------------------------------------------


def test_it_starts_in_swinging_so_the_first_thing_it_seeks_is_digging():
    """Starting anywhere else would mean guessing what the machine was doing
    before the clip began. Starting in swinging costs nothing: the leading
    partial cycle is discarded either way, because a cycle is dig-onset to
    dig-onset."""
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    assert state.curr_stage == "swinging"
    assert state.looking_for == "digging"


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("swinging", "digging"),
        ("digging", "hauling"),
        ("hauling", "dumping"),
        ("dumping", "swinging"),
    ],
)
def test_looking_for_walks_the_fixed_cycle(current, expected):
    from excavator_cycles.fsm import MachineState

    assert MachineState(curr_stage=current).looking_for == expected


def test_advancing_records_the_onset_and_moves_on():
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    state.advance("digging", 4.2)
    assert state.curr_stage == "digging"
    assert state.since == 4.2
    assert state.pending["digging"] == 4.2
    assert state.looking_for == "hauling"


def test_advancing_out_of_order_is_refused():
    """Ordering is meant to be impossible to violate, not merely checked later.
    If this ever passes silently, the walk has a bug that would show up as a
    nonsense duration rather than as an error."""
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    with pytest.raises(ValueError, match="looking for digging"):
        state.advance("dumping", 4.2)


def test_time_must_move_forward():
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    state.advance("digging", 4.2)
    with pytest.raises(ValueError, match="backwards"):
        state.advance("hauling", 3.0)


def test_a_phase_can_be_noted_as_having_happened_without_an_onset():
    """The weak second check. This is what separates "we missed a cue" from "no
    cycle happened" when a cycle is closed."""
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    state.note_occurred("dumping")
    assert "dumping" in state.occurred
    assert "dumping" not in state.pending, "occurring is not the same as being located"


def test_elapsed_reports_how_long_we_have_been_in_this_phase():
    from excavator_cycles.fsm import MachineState

    state = MachineState()
    state.advance("digging", 4.0)
    assert state.elapsed(6.5) == pytest.approx(2.5)


def test_elapsed_before_anything_has_started_is_none():
    from excavator_cycles.fsm import MachineState

    assert MachineState().elapsed(3.0) is None
