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


# --- the walk -------------------------------------------------------------
#
# The trigger is injected throughout. The walk's job is sequencing, validation
# and recovery; whether a cue is any good is a separate question, tested
# separately. Mixing the two would mean a cue change could break a sequencing
# test for reasons that have nothing to do with sequencing.


def _walk_table(n=200):
    return _Table(height=np.zeros(n), truck_overlap=np.zeros(n), speed_x=np.zeros(n))


def _scripted(schedule):
    """A trigger that fires for `phase` exactly on the sample indices given.

    schedule: {phase: set_of_indices}
    """

    def fires(phase, table, index, levels, config):
        return index in schedule.get(phase, set())

    return fires


def _levels():
    from excavator_cycles.fsm import Levels, Split

    clean = Split(0.5, 0.95, 100, 100)
    return Levels(low_height=clean, over_truck=clean, moving=clean)


def test_a_clean_run_of_triggers_produces_onsets_in_order():
    from excavator_cycles.fsm import walk

    hold = 3
    schedule = {
        "digging": set(range(10, 10 + hold)),
        "hauling": set(range(40, 40 + hold)),
        "dumping": set(range(70, 70 + hold)),
        "swinging": set(range(100, 100 + hold)),
    }
    found = walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=hold)
    assert [d.phase for d in found] == ["digging", "hauling", "dumping", "swinging"]


def test_three_cycles_are_walked_without_anything_written_per_cycle():
    """Multi-cycle is meant to fall out of the loop, not be special-cased."""
    from excavator_cycles.fsm import walk

    hold, schedule = 3, {}
    for cycle in range(3):
        base = 10 + cycle * 60
        for offset, phase in enumerate(("digging", "hauling", "dumping", "swinging")):
            schedule.setdefault(phase, set()).update(
                range(base + offset * 12, base + offset * 12 + hold)
            )
    found = walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=hold)
    assert [d.phase for d in found].count("digging") == 3
    assert len(found) == 12


def test_a_single_noisy_sample_does_not_advance_the_state():
    """Validation is the whole reason a trigger is not a transition."""
    from excavator_cycles.fsm import walk

    schedule = {"digging": {10}}  # fires once, then stops
    assert walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=3) == []


def test_evidence_held_long_enough_does_advance():
    from excavator_cycles.fsm import walk

    schedule = {"digging": set(range(10, 13))}
    found = walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=3)
    assert len(found) == 1 and found[0].phase == "digging"


def test_the_onset_is_credited_to_where_the_evidence_STARTED():
    """Validation costs latency -- you only believe it after k samples -- and
    that latency must not be baked into the answer."""
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 13))}),
        hold_samples=3,
    )
    assert found[0].fired_at == 10, "credited to the confirming sample, not the first"


def test_the_coarse_window_reaches_back_before_the_trigger():
    """An onset is where a signal LEFT rest, which is found by walking BACKWARD
    from the excursion. A window starting at the trigger would exclude it."""
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(30, 33))}),
        hold_samples=3,
        lookback_samples=8,
    )
    assert found[0].window.lo <= 30 - 8
    assert found[0].window.hi >= 33


def test_a_window_cannot_reach_back_past_the_start_of_the_clip():
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(1, 4))}),
        hold_samples=3,
        lookback_samples=50,
    )
    assert found[0].window.lo == 0


def test_digging_is_watched_even_when_we_are_looking_for_something_else():
    """Digging is the cycle boundary, so it is the only way back from
    confusion. Watched only in sequence, one missed onset loses every later
    cycle."""
    from excavator_cycles.fsm import walk

    hold = 3
    schedule = {
        # the second dig is out of turn, so it must clear the HIGHER bar: the
        # default strict hold is twice `hold`, hence six samples, not three
        "digging": set(range(10, 13)) | set(range(50, 56)),
        "hauling": set(range(20, 23)),
        # dumping never fires -- the machine re-dug instead
    }
    found = walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=hold)
    phases = [d.phase for d in found]
    assert phases == ["digging", "hauling", "digging"], phases
    assert found[-1].out_of_sequence, "the second dig interrupted hauling"


def test_an_out_of_sequence_dig_must_clear_a_higher_bar():
    """Expected evidence is cheap; unexpected evidence should be expensive. A
    spurious dig mid-haul would silently truncate a perfectly good cycle."""
    from excavator_cycles.fsm import walk

    schedule = {
        "digging": set(range(10, 13)) | set(range(50, 53)),  # only `hold`, not strict
        "hauling": set(range(20, 23)),
    }
    found = walk(_walk_table(), _levels(), fires=_scripted(schedule), hold_samples=3)
    assert [d.phase for d in found] == ["digging", "hauling"], (
        "three samples is enough in sequence and must not be enough out of it"
    )


def test_an_in_sequence_dig_is_not_marked_out_of_sequence():
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 13))}),
        hold_samples=3,
    )
    assert not found[0].out_of_sequence
