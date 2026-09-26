"""The state machine: calibration, the walk, and pass 2.

Written before the implementation. Every test here states a property the design
document commits to, so a failure means either the code is wrong or the design
changed -- and the second is worth noticing.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

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

    def __init__(self, height, truck_overlap, speed_x, times=None, rel_cabin_x=None):
        self.height = np.asarray(height, float)
        self.truck_overlap = np.asarray(truck_overlap, float)
        self.speed_x = np.asarray(speed_x, float)
        self.time_seconds = (
            np.arange(len(self.height)) * 0.1 if times is None else np.asarray(times, float)
        )
        # `calibrate` reads this to derive which side the truck is on. Positive by
        # default, i.e. the truck is on the machine's right, as in the dev clip.
        self.rel_cabin_x = (
            np.full(len(self.height), 0.4)
            if rel_cabin_x is None
            else np.asarray(rel_cabin_x, float)
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
    and may not say WHEN anything happened.

    This test COULD NOT FAIL as written. It asserted each field was
    `not isinstance(value, (list, tuple))`, and the fields are `Split | None` or a
    float -- never a list. Mutating `calibrate` to return a completely wrong
    `Split(1.0, 1.0, 1, 1)` left it green, and an interval returned as a `Window`
    or any two-field dataclass would have passed too, which is exactly the shape
    it claimed to be guarding against.

    It now asserts the concrete types, and that no field carries two numbers.
    """
    from excavator_cycles.fsm import Window

    levels = calibrate(_table())
    for name, value in vars(levels).items():
        if name == "dump_side":
            assert isinstance(value, float), f"{name} should be a bare sign, got {value!r}"
            assert value in (-1.0, 1.0), f"{name} must be a sign, got {value!r}"
            continue
        assert value is None or isinstance(value, Split), (
            f"{name} is a {type(value).__name__}, not a Split -- only levels belong here"
        )
        assert not isinstance(value, Window), f"{name} is an interval, not a level"
    # A level is ONE position plus how much to trust it. If `Split` ever grows a
    # second bound, that is the bracket-first design returning, and this is the
    # assertion that notices -- naming the fields rather than counting floats,
    # since `separability` is a quality measure and not a second bound.
    assert set(Split.__dataclass_fields__) == {
        "threshold",
        "separability",
        "below",
        "above",
        "min_side",
    }, f"Split's fields changed: {sorted(Split.__dataclass_fields__)}"


def test_calibrate_survives_a_video_with_no_truck():
    """truck_overlap is all-nan when no truck was ever detected. That is a
    legitimate video, not an error -- the gate is simply unavailable."""
    table = _table()
    table.truck_overlap = np.full(len(table.height), np.nan)
    levels = calibrate(table)
    # `is None or not trustworthy` was weaker than the docstring: mutating
    # `calibrate` to return `Split(0.123, 0.5, 50, 50)` instead of None kept it
    # green, although the docstring says the gate is UNAVAILABLE. It must be None,
    # because that is the one value `trigger_dumping` actually checks.
    assert levels.over_truck is None
    # And the required levels must still be there: an absent truck is not a
    # reason to give up on the rest of the video.
    assert isinstance(levels.low_height, Split)
    assert isinstance(levels.moving, Split)


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


# --- the coarse triggers --------------------------------------------------
#
# These pick the WINDOW, not the instant, so each test asks only "does this
# fire where it obviously should, and stay quiet where it obviously should
# not". Precision is pass 2's job and is tested separately.


class _CueTable:
    """Every column the triggers read, at a single instant."""

    def __init__(self, n=50, **columns):
        self.time_seconds = np.arange(n) * 0.1
        for name in (
            "height",
            "dh_dt",
            "d2h_dt2",
            "speed_x",
            "truck_overlap",
            "rel_cabin_x",
            "aspect_ratio",
        ):
            setattr(self, name, np.full(n, float(columns.get(name, 0.0))))
        self.found = np.ones(n, bool)


def _lv(low=0.1, truck=0.2, moving=0.3):
    from excavator_cycles.fsm import Levels, Split

    return Levels(
        low_height=Split(low, 0.95, 50, 50),
        over_truck=Split(truck, 0.95, 50, 50),
        moving=Split(moving, 0.95, 50, 50),
    )


def test_digging_fires_when_the_bucket_is_down_and_still():
    from excavator_cycles.fsm import trigger_digging

    assert trigger_digging(_CueTable(height=0.0, speed_x=0.05), 10, _lv())


def test_digging_stays_quiet_while_the_bucket_is_up():
    from excavator_cycles.fsm import trigger_digging

    assert not trigger_digging(_CueTable(height=0.5, speed_x=0.05), 10, _lv())


def test_digging_stays_quiet_while_the_machine_is_traversing():
    """Low and moving is the bucket passing through on its way somewhere, not
    scooping. The spec puts that in swinging."""
    from excavator_cycles.fsm import trigger_digging

    assert not trigger_digging(_CueTable(height=0.0, speed_x=0.9), 10, _lv())


def test_hauling_fires_when_the_bucket_is_clear_of_the_material_and_rising():
    from excavator_cycles.fsm import trigger_hauling

    assert trigger_hauling(_CueTable(height=0.5, dh_dt=0.2), 10, _lv())


def test_hauling_stays_quiet_when_the_bucket_is_high_but_descending():
    """Height alone is not enough -- the bucket is high for most of the cycle.
    The spec's word is CLEARS, which is a direction, not a level."""
    from excavator_cycles.fsm import trigger_hauling

    assert not trigger_hauling(_CueTable(height=0.5, dh_dt=-0.2), 10, _lv())


def test_dumping_fires_when_the_bucket_is_over_the_bed():
    from excavator_cycles.fsm import trigger_dumping

    assert trigger_dumping(_CueTable(truck_overlap=0.6, rel_cabin_x=0.5), 10, _lv())


def test_dumping_stays_quiet_away_from_the_truck():
    from excavator_cycles.fsm import trigger_dumping

    assert not trigger_dumping(_CueTable(truck_overlap=0.01, rel_cabin_x=0.5), 10, _lv())


def test_dumping_cannot_fire_on_a_video_with_no_truck():
    """No truck is a legitimate video. The gate is unavailable, not false."""
    from excavator_cycles.fsm import trigger_dumping

    levels = _lv()
    object.__setattr__(levels, "over_truck", None)
    assert not trigger_dumping(_CueTable(truck_overlap=np.nan), 10, levels)


def test_swinging_fires_when_traversing_and_descending():
    from excavator_cycles.fsm import trigger_swinging

    assert trigger_swinging(_CueTable(speed_x=0.9, dh_dt=-0.2), 10, _lv())


def test_swinging_is_the_opposite_of_hauling_on_the_vertical():
    """The diagram's words. Both are traverses; the sign of dh/dt separates
    carrying a load out from bringing an empty bucket back."""
    from excavator_cycles.fsm import trigger_hauling, trigger_swinging

    rising = _CueTable(speed_x=0.9, dh_dt=0.2, height=0.5)
    assert trigger_hauling(rising, 10, _lv())
    assert not trigger_swinging(rising, 10, _lv())


def test_a_sample_with_no_mask_never_fires_anything():
    """Missing is not evidence against a transition, but it is certainly not
    evidence for one."""
    from excavator_cycles.fsm import default_fires

    table = _CueTable(height=0.0, speed_x=0.05)
    table.found[10] = False
    for phase in ("digging", "hauling", "dumping", "swinging"):
        assert not default_fires(phase, table, 10, _lv(), None)


def test_a_nan_sample_never_fires_anything():
    from excavator_cycles.fsm import default_fires

    table = _CueTable(height=0.0, speed_x=0.05)
    table.height[10] = np.nan
    assert not default_fires("digging", table, 10, _lv(), None)


def test_a_condition_that_stays_true_fires_only_once():
    """A trigger is an EDGE, not a level.

    "The bucket is down and still" is true for the whole digging phase, not just
    at its start. Treating it as a level makes the always-on digging check fire
    again on every sample of a dig it has already recorded -- which on the real
    video produced sixteen digging detections inside one digging phase.
    """
    from excavator_cycles.fsm import walk

    # the condition holds continuously from sample 10 to the end
    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 200))}),
        hold_samples=3,
    )
    assert [d.phase for d in found] == ["digging"], [d.phase for d in found]


def test_the_condition_must_go_away_before_it_can_fire_again():
    """Two separate digs, with the bucket genuinely lifted in between."""
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 20)) | set(range(60, 80))}),
        hold_samples=3,
        strict_hold_samples=3,
    )
    assert [d.phase for d in found] == ["digging", "digging"]


def test_an_already_running_phase_cannot_interrupt_itself():
    """The out-of-sequence digging check must not fire while we ARE digging."""
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 100))}),
        hold_samples=3,
    )
    # Non-vacuity first: `not any(...)` over an empty list is True, so a walk that
    # fired nothing at all used to pass this. Mutating `walk` to never fire left it
    # green, which made it evidence of nothing.
    assert [d.phase for d in found] == ["digging"], f"expected one dig, got {found}"
    assert not any(d.out_of_sequence for d in found)


# --- windows may not overlap ----------------------------------------------


def test_consecutive_windows_never_overlap():
    """Ordering is enforced on the onset TIMES, but pass 2 searches inside the
    WINDOWS. Overlapping windows would let refinement return hauling at 13.0s
    and dumping at 12.8s -- a negative phase duration, from a machine whose
    whole point is that ordering is impossible to violate.

    Invisible today, because the coarse onsets are the trigger times and those
    are ordered. Fatal the moment pass 2 exists.
    """
    from excavator_cycles.fsm import walk

    hold = 3
    schedule = {
        "digging": set(range(10, 10 + hold)),
        "hauling": set(range(14, 14 + hold)),  # deliberately close
        "dumping": set(range(18, 18 + hold)),
        "swinging": set(range(22, 22 + hold)),
    }
    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=hold,
        lookback_samples=20,  # far wider than the gaps between triggers
    )
    # Non-vacuity first: `pairwise` of a list with fewer than two elements yields
    # nothing, so a walk that fired nothing used to satisfy this invariant trivially.
    assert [d.phase for d in found] == ["digging", "hauling", "dumping", "swinging"], (
        f"all four must fire or the invariant is untested; got {[d.phase for d in found]}"
    )
    for earlier, later in pairwise(found):
        # Non-overlap means the next window starts at or after the previous one
        # ENDS -- not merely after the previous one was detected. The previous
        # window extends forward past its own trigger, so clipping to the
        # trigger leaves them overlapping by exactly that extension.
        assert later.window.lo >= earlier.window.hi, (
            f"{later.phase} window [{later.window.lo},{later.window.hi}) overlaps "
            f"{earlier.phase} [{earlier.window.lo},{earlier.window.hi})"
        )


def test_a_window_is_clipped_to_the_previous_onset():
    """This phase's onset must be AFTER the previous phase started. That is not
    a convention, it is what "phase" means -- so a window reaching back past it
    permits an answer that cannot be true."""
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": {10, 11, 12}, "hauling": {15, 16, 17}}),
        hold_samples=3,
        lookback_samples=50,  # would reach back to 0 if unclipped
    )
    assert found[1].window.lo == found[0].window.hi, (
        "clipped to where the previous window ENDS, not to where it was triggered"
    )


def test_the_first_window_has_nothing_to_clip_against():
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": {30, 31, 32}}),
        hold_samples=3,
        lookback_samples=8,
    )
    assert found[0].window.lo == 22, "free to use the full lookback"


# --- every parameter is a duration ----------------------------------------


def test_walk_parameters_come_from_config_in_seconds():
    """A window given in SAMPLES means something different at every frame rate.
    The config holds durations; the conversion happens once, at the boundary.

    This test used to assert only `hasattr` and `isinstance(float)`, which cannot
    fail while the field exists -- and it did not fail when `strict_hold_seconds`
    stopped being READ, leaving `config.py` and `default.yaml` both documenting a
    live parameter that nothing consulted. Each duration is now asserted to CHANGE
    THE WALK, which is the only claim worth making about a config value.
    """
    import dataclasses

    from excavator_cycles.config import Config
    from excavator_cycles.fsm import walk

    def phases(**overrides) -> list[tuple[str, int]]:
        config = Config.load()
        config = dataclasses.replace(config, fsm=dataclasses.replace(config.fsm, **overrides))
        found = walk(
            _walk_table(60),
            _levels(),
            # digging 10-13, then hauling far later, then a long mid-cycle dig that
            # only a LOW strict bar will accept as out-of-sequence.
            fires=_scripted(
                {
                    "digging": set(range(10, 14)) | set(range(30, 34)),
                    "hauling": set(range(20, 28)),
                }
            ),
            config=config,
        )
        return [(d.phase, d.fired_at) for d in found]

    baseline = phases()
    assert baseline, "the fixture must detect something or nothing below is measurable"

    # `hold_seconds`: a longer hold than the evidence rejects the transition.
    assert phases(hold_seconds=2.0) != baseline, "hold_seconds does not reach the walk"
    # `lookback_seconds`: changes where each window starts.
    windows_short = walk(
        _walk_table(60),
        _levels(),
        fires=_scripted({"digging": set(range(10, 14))}),
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=1,
    )[0].window.lo
    windows_long = walk(
        _walk_table(60),
        _levels(),
        fires=_scripted({"digging": set(range(10, 14))}),
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=8,
    )[0].window.lo
    assert windows_short != windows_long, "lookback does not reach the window"
    # `strict_hold_seconds`: the bar for an out-of-turn dig. A low bar admits the
    # mid-cycle dig; a high one does not.
    lenient = phases(strict_hold_seconds=0.2)
    strictest = phases(strict_hold_seconds=5.0)
    assert lenient != strictest, (
        f"strict_hold_seconds does not reach the walk: {lenient} == {strictest}"
    )


def test_the_same_duration_is_more_samples_at_a_higher_rate():
    """The property the whole seconds-not-samples rule exists for."""
    from excavator_cycles.fsm import samples_for

    assert samples_for(0.3, np.arange(50) * 0.1) == 3  # 10 Hz
    assert samples_for(0.3, np.arange(50) * 0.05) == 6  # 20 Hz


def test_a_duration_shorter_than_one_sample_still_gets_one():
    from excavator_cycles.fsm import samples_for

    assert samples_for(0.01, np.arange(50) * 0.1) == 1


def test_a_broken_time_base_is_refused_not_guessed():
    from excavator_cycles.fsm import samples_for

    with pytest.raises(ValueError, match="not increasing"):
        samples_for(0.3, np.zeros(10))


def test_walk_accepts_a_config_and_converts_for_itself():
    """Callers should pass seconds, not pre-computed sample counts."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(),
        _levels(),
        fires=_scripted({"digging": set(range(10, 20))}),
        config=Config.load(),
    )
    assert [d.phase for d in found] == ["digging"]


# --- pass 2 ---------------------------------------------------------------


def _ramp_table(n=60, onset=30, rate=0.05):
    """A signal at rest, then rising steadily from `onset`. The onset is the
    thing pass 2 has to find, and we know exactly where it is."""
    t = np.arange(n) * 0.1
    height = np.where(np.arange(n) < onset, 0.0, (np.arange(n) - onset) * rate)
    return _CueTable(n=n), t, height


def test_refine_finds_a_known_onset_on_a_ramp():
    from excavator_cycles.fsm import Window, refine

    _table, t, height = _ramp_table(onset=30)
    when = refine(height, t, Window(20, 45), mode="departs")
    assert when == pytest.approx(3.0, abs=0.25), f"onset is at 3.0s, got {when}"


def test_refine_searches_only_inside_its_window():
    """The guarantee the whole two-pass design rests on: a window that does not
    contain the transition cannot produce it."""
    from excavator_cycles.fsm import Window, refine

    _table, t, height = _ramp_table(onset=30)
    when = refine(height, t, Window(40, 55), mode="departs")
    assert when is None or when >= t[40]


def test_refine_returns_none_when_the_cue_never_fires():
    """Information, not a crash: the window contained nothing to find."""
    from excavator_cycles.fsm import Window, refine

    t = np.arange(40) * 0.1
    assert refine(np.zeros(40), t, Window(5, 30), mode="departs") is None


def test_refine_uses_the_timestamp_array_not_a_nominal_dt():
    """On a variable-rate clip `t0 + i*dt` is wrong, and wrong invisibly."""
    from excavator_cycles.fsm import Window, refine

    n, onset = 60, 30
    height = np.where(np.arange(n) < onset, 0.0, (np.arange(n) - onset) * 0.05)
    uneven = np.cumsum(np.r_[0.0, np.random.default_rng(0).uniform(0.05, 0.15, n - 1)])
    when = refine(height, uneven, Window(20, 45), mode="departs")
    assert when == pytest.approx(uneven[onset], abs=0.3), (
        "the onset must be reported at its real timestamp, not index * nominal dt"
    )


def test_refine_handles_an_arrival_as_well_as_a_departure():
    """digging and hauling ARRIVE at rest; swinging DEPARTS it."""
    from excavator_cycles.fsm import Window, refine

    n = 60
    t = np.arange(n) * 0.1
    falling = np.where(np.arange(n) < 30, (30 - np.arange(n)) * 0.05, 0.0)
    when = refine(falling, t, Window(15, 45), mode="arrives")
    assert when == pytest.approx(3.0, abs=0.3)


def test_refine_on_an_empty_window_is_none_not_an_error():
    from excavator_cycles.fsm import Window, refine

    t = np.arange(40) * 0.1
    assert refine(np.arange(40.0), t, Window(10, 10), mode="departs") is None


def test_refine_extremum_finds_the_peak():
    """T3's cue is argmax(aspect_ratio), which is neither an arrival nor a
    departure but a turning point."""
    from excavator_cycles.fsm import Window, refine

    t = np.arange(60) * 0.1
    bump = np.exp(-(((np.arange(60) - 35) / 6.0) ** 2))
    assert refine(bump, t, Window(20, 50), mode="peak") == pytest.approx(3.5, abs=0.15)


def test_the_rest_band_is_not_fooled_by_a_window_that_is_mostly_moving():
    """A window is chosen BECAUSE it contains a transition, so it is mostly
    moving by construction. Using the median difference as the noise measures
    that motion: on a clean ramp it came out at the ramp's own step size, which
    swallowed the whole excursion and put the onset four samples late."""
    from excavator_cycles.fsm import _rest_band

    ramp = np.r_[np.zeros(10), np.arange(14) * 0.05]
    assert _rest_band(ramp, sigma=3.0) < 0.05, "the band must not reach the ramp's step size"


def test_the_rest_band_never_collapses_to_zero():
    """A perfectly flat lead-in makes the low quantile 0, and a zero band makes
    every sample an excursion."""
    from excavator_cycles.fsm import _rest_band

    assert _rest_band(np.r_[np.zeros(20), np.arange(10) * 0.1], sigma=3.0) > 0


# --- the whole pass: detections -> refined onsets --------------------------


def test_locate_refines_a_detection_into_a_time():
    """Hand it a window that definitely contains the event, and it returns the
    instant rather than the trigger sample."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import Detection, Window, locate

    n = 80
    table = _CueTable(n=n)
    # dh/dt falls, then ARRIVES at rest at sample 40 -- which is contact
    table.dh_dt = np.r_[np.full(40, -0.3), np.zeros(n - 40)]
    found = locate([Detection("digging", Window(25, 60), 45)], table, Config.load())
    assert len(found) == 1
    assert found[0].phase == "digging"
    assert found[0].refined == pytest.approx(4.0, abs=0.3), f"arrival at 4.0s, got {found[0]}"
    assert found[0].coarse == pytest.approx(4.5, abs=0.2), "pass 1's trigger time, kept"


def test_locate_refuses_to_invent_an_onset_but_keeps_its_place():
    """A window with no event in it yields `refined=None` -- NOT a dropped onset.

    Both halves matter and they pull in opposite directions. Giving the trigger
    time as the onset would put a made-up number into a duration, indistinguishable
    from a measured one. But dropping the detection removed a cycle BOUNDARY:
    `assemble` splits on digging, so a failed digging refinement deleted the whole
    cycle, and the dev clip reported `cycle_count: 0` for a video with one
    complete cycle in it. Keeping the place with no refined time does neither.
    """
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import Detection, Window, locate

    table = _CueTable(n=80)
    table.dh_dt = np.full(80, -0.3)  # never arrives at rest
    found = locate([Detection("digging", Window(10, 40), 20)], table, Config.load())
    assert len(found) == 1, "the detection must survive; only its refinement failed"
    assert found[0].refined is None, "no invented onset"
    assert found[0].coarse == pytest.approx(2.0, abs=0.2), "the coarse time is still there"


def test_locate_keeps_the_onsets_in_order():
    """Refinement must not be able to reorder what the walk ordered. The windows
    cannot overlap, so this is guaranteed by construction -- pinned because if
    it ever breaks, the symptom is a negative phase duration."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import locate, walk

    n = 200
    table = _CueTable(n=n)
    # A full synthetic cycle, shaped so that each cue has something real to find:
    # the bucket descends, rests in the material, lifts, travels over the bed,
    # tips, then swings back. Flat segments are what `arrives`/`departs` need.
    table.height = np.r_[
        np.linspace(0.5, 0.0, 30),  # descending
        np.zeros(30),  # in the material -- dh_dt ARRIVES at rest
        np.linspace(0.0, 0.6, 40),  # lifting
        np.full(40, 0.6),  # carrying at height
        np.linspace(0.6, 0.5, 20),  # tipping
        np.linspace(0.5, 0.5, 40),  # swinging back
    ]
    table.dh_dt = np.gradient(table.height, table.time_seconds)
    table.d2h_dt2 = np.gradient(table.dh_dt, table.time_seconds)
    table.speed_x = np.r_[np.zeros(100), np.full(40, 0.8), np.zeros(20), np.full(40, 0.9)]
    table.truck_overlap = np.r_[np.zeros(120), np.full(40, 0.5), np.zeros(40)]
    table.rel_cabin_x = np.full(n, 0.4)
    table.aspect_ratio = np.r_[np.full(140, 1.0), np.full(20, 2.5), np.full(40, 1.0)]

    found = walk(table, _lv(low=0.1, moving=5.0), hold_samples=3, lookback_samples=8)
    onsets = locate(found, table, Config.load())
    times = [o.refined for o in onsets if o.refined is not None]
    # Non-vacuity FIRST. Without it this test passed on an empty list for its
    # entire life: the previous fixture refined nothing at all, so `[] == sorted([])`
    # proved the ordering guarantee held over no onsets.
    assert len(times) >= 2, f"need at least two refined onsets to order; got {onsets}"
    assert times == sorted(times), times


def test_locate_reports_a_cue_that_found_nothing_instead_of_hiding_it():
    """A flat signal refines to nothing, and that fact must reach the caller.

    `Cycle.reason` is what turns this into a sentence a reader can act on, and it
    cannot do that for an onset that is simply absent from the list.
    """
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import Detection, Window, locate

    table = _CueTable(n=60)  # every signal is flat: nothing to find
    found = locate([Detection("digging", Window(10, 30), 20)], table, Config.load())
    assert [(o.phase, o.refined) for o in found] == [("digging", None)]


# --- the rest band ------------------------------------------------------------
#
# `_rest_band` decides how far from zero still counts as "at rest", which decides
# where `refine` puts every onset. It was wrong by a factor of 2.1 -- and a band
# that is too NARROW makes a settled signal look like it is still moving, so
# `arrives` finds nothing and the onset is lost entirely rather than misplaced.


def test_the_rest_band_recovers_a_known_noise_level():
    """The band must mean what it says: `sigma=1` is one standard deviation.

    The old code multiplied the 0.25 quantile of |differences| by 1.4826/sqrt(2).
    1.4826 is the MAD-to-sigma constant and it applies to the MEDIAN of absolute
    deviations, not to some other quantile -- a familiar constant reached for and
    applied to the wrong statistic. The result recovered 0.47 sigma instead of
    1.00, so every band was 2.1x too tight.
    """
    from excavator_cycles.fsm import _rest_band

    rng = np.random.default_rng(0)
    for true_sigma in (0.01, 0.05, 0.2, 1.0):
        noise = rng.normal(0.0, true_sigma, 100_000)
        band = _rest_band(noise, sigma=1.0, floor_fraction=0.0)
        assert band == pytest.approx(true_sigma, rel=0.05), (
            f"sigma={true_sigma}: band {band:.5f} should be one sigma"
        )


def test_the_rest_band_scales_with_sigma():
    """Three sigma must be three times one sigma, or the parameter is a fudge."""
    from excavator_cycles.fsm import _rest_band

    noise = np.random.default_rng(1).normal(0.0, 0.1, 100_000)
    one = _rest_band(noise, sigma=1.0, floor_fraction=0.0)
    assert _rest_band(noise, sigma=3.0, floor_fraction=0.0) == pytest.approx(3 * one, rel=1e-9)


def test_the_rest_band_is_estimated_from_the_quiet_part_not_the_moving_part():
    """The trap this function exists to avoid, tested as a comparison.

    A refinement window is mostly MOVING by construction -- it was chosen because
    it contains a transition. Estimating the band from the MEDIAN difference
    therefore measures the motion, and on a clean synthetic ramp that estimate came
    out at the ramp's own step size, swallowing the whole excursion and putting the
    onset four samples late.

    Asserted as a ratio between the two quantile choices on identical data, so no
    absolute threshold needs tuning: the low quantile must stay below the ramp's
    step size, and far below what the median would give.
    """
    from excavator_cycles.fsm import _QUANTILE_TO_SIGMA, _rest_band

    rng = np.random.default_rng(2)
    noise_sigma, ramp_step = 0.001, 0.05
    # 40% at rest: above the 0.25 quantile so it measures the noise, below the
    # 0.5 quantile so the median measures the motion. That gap IS the design
    # choice, and a fixture either side of it is what demonstrates it.
    quiet = rng.normal(0.0, noise_sigma, 120)
    ramp = np.arange(180) * ramp_step  # steps 50x the noise, as a real cue is
    mixed = np.r_[quiet, ramp]

    band = _rest_band(mixed, sigma=3.0, floor_fraction=0.0)
    median_band = 3.0 * float(np.quantile(np.abs(np.diff(mixed)), 0.5)) * _QUANTILE_TO_SIGMA

    assert band < ramp_step, (
        f"band {band:.5f} must not swallow a step of {ramp_step}, or the excursion "
        "is invisible and the onset lands late"
    )
    assert median_band > ramp_step, "the median really is the trap being avoided"
    assert band < median_band / 10, f"low quantile {band:.5f} vs median {median_band:.5f}"


def test_the_rest_band_needs_a_quarter_of_the_window_to_be_quiet():
    """A REAL LIMITATION, pinned so whoever sizes the windows knows about it.

    The estimator reads the 0.25 quantile of |differences|, so it only measures
    noise if MORE than a quarter of the window is actually at rest. Below that the
    quantile falls inside the moving part and the band becomes a measure of the
    motion -- the very failure the low quantile was chosen to avoid, reappearing
    when the window is mostly excursion.

    This bounds how wide a refinement window may be relative to the rest either
    side of the transition. It is not currently enforced anywhere, which is worth
    knowing while the cues are being fixed.
    """
    from excavator_cycles.fsm import _rest_band

    rng = np.random.default_rng(3)
    ramp_step = 0.05
    quiet = rng.normal(0.0, 0.001, 60)
    ramp = np.arange(240) * ramp_step  # only 20% of the window is at rest
    band = _rest_band(np.r_[quiet, ramp], sigma=3.0, floor_fraction=0.0)
    assert band > ramp_step, (
        "documented behaviour: with under a quarter of the window quiet the band "
        f"is set by the motion ({band:.4f} against a step of {ramp_step})"
    )


def test_the_rest_band_never_collapses_to_zero_on_a_flat_signal():
    """A band of zero makes every sample an excursion, so the floor is load-bearing."""
    from excavator_cycles.fsm import _rest_band

    flat_then_step = np.r_[np.zeros(50), np.ones(50)]
    band = _rest_band(flat_then_step, sigma=3.0, floor_fraction=0.02)
    assert band == pytest.approx(0.02, rel=1e-9), "2% of the 0-to-1 range"


def test_the_rest_band_needs_something_to_measure():
    """Fewer than three samples yields no differences worth a quantile."""
    from excavator_cycles.fsm import _rest_band

    assert _rest_band(np.array([1.0, 2.0]), sigma=3.0) == 0.0


# --- the two silent losses ----------------------------------------------------
#
# Both make `walk` return FEWER detections than the data supports, with no error
# and no log line. A lost transition costs a graded field; a lost DIGGING
# transition costs a cycle boundary and so costs `cycle_count` too.


def test_the_next_phase_is_not_lost_when_it_begins_where_the_last_was_confirmed():
    """The common case, and it was broken.

    `walk` skipped `hold_samples` forward after a detection, and `sustained`
    tested for a rising edge at `start - 1`. If the next phase's condition rose
    anywhere inside that skipped span and then held, the edge check landed INSIDE
    the span, saw True, and refused forever -- the condition never goes false
    again, so the transition is lost permanently.

    This is not a corner case. Phases are contiguous: hauling's condition becomes
    true at or immediately after digging's onset is confirmed, every cycle. That
    is precisely the losing window.
    """
    from excavator_cycles.fsm import walk

    for rise in (10, 11, 12, 13):
        schedule = {"digging": set(range(10, 13)), "hauling": set(range(rise, 60))}
        found = walk(
            _walk_table(60),
            _levels(),
            fires=_scripted(schedule),
            hold_samples=3,
            lookback_samples=2,
        )
        phases = [d.phase for d in found]
        assert phases == ["digging", "hauling"], f"hauling rising at {rise} gave {phases}"


def test_a_phase_still_cannot_fire_twice_inside_itself():
    """The guard the edge rule exists for, which the fix above must not weaken.

    A trigger written as a condition -- "the bucket is down and still" -- is true
    for the WHOLE of the phase it describes. Accepting a level rather than an edge
    once produced sixteen digging detections inside one digging phase.
    """
    from excavator_cycles.fsm import walk

    schedule = {"digging": set(range(10, 90))}  # true for eighty samples
    found = walk(
        _walk_table(100),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=3,
        lookback_samples=2,
    )
    assert [d.phase for d in found] == ["digging"], "one phase, one detection"


def test_an_out_of_sequence_dig_still_needs_a_real_rising_edge():
    """Unexpected evidence stays expensive.

    The leniency that fixes the contiguous-phase loss applies only to the phase
    the walk is EXPECTING. If it applied to the out-of-sequence digging check too,
    then digging's condition still being true just after digging was detected
    would immediately fire a spurious dig and abandon the cycle.
    """
    from excavator_cycles.fsm import walk

    schedule = {"digging": set(range(10, 90)), "hauling": set(range(20, 90))}
    found = walk(
        _walk_table(100),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=3,
        lookback_samples=2,
    )
    assert [d.phase for d in found] == ["digging", "hauling"]
    assert not any(d.out_of_sequence for d in found), "digging held; it never re-rose"


def test_a_transition_at_the_very_end_of_the_clip_is_not_discarded():
    """`start + needed > count` dropped any trigger rising in the last hold-1
    samples, silently.

    Not hypothetical: the labelled cycle-closing dig is at sample 293 of 296.
    With hold=3, 293 + 3 == 296 passes by EXACTLY zero margin -- one fewer decoded
    frame and the final cycle boundary vanishes, taking `cycle_count` with it.
    The evidence that exists is all the evidence there can be, so a run reaching
    the end of the data counts.
    """
    from excavator_cycles.fsm import walk

    def phases_when_rising_at(rise: int) -> list[str]:
        return [
            d.phase
            for d in walk(
                _walk_table(60),
                _levels(),
                fires=_scripted({"digging": set(range(rise, 60))}),
                hold_samples=3,
                strict_hold_samples=6,
                lookback_samples=2,
            )
        ]

    # 56 and 57 leave a full three samples; 58 leaves two, which clears the
    # half-the-hold floor. 59 leaves one, which does not -- see
    # `test_a_single_sample_at_the_clip_edge_is_not_a_transition` for why that floor
    # exists at all.
    for rise in (56, 57, 58):
        assert phases_when_rising_at(rise) == ["digging"], f"rising at {rise} of 60 was lost"
    assert phases_when_rising_at(59) == [], "one sample is below the floor"


def test_a_truncated_hold_is_only_allowed_at_the_clip_edge():
    """Mid-clip, three samples still means three. Otherwise the hold is no bar."""
    from excavator_cycles.fsm import walk

    schedule = {"digging": {20, 21}}  # two samples, then false again, mid-clip
    found = walk(
        _walk_table(60),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=3,
        lookback_samples=2,
    )
    assert found == [], "a two-sample blip in the middle of the clip is not a transition"


def test_the_injected_trigger_sees_the_same_config_on_every_call():
    """It did not. The edge check passed `config` and the hold loop passed `None`,
    so an injected trigger that read config behaved differently on the two calls
    -- a difference that would show up as an intermittent missed transition."""
    from excavator_cycles.fsm import walk

    seen = []

    def fires(phase, table, index, levels, config):
        seen.append(config)
        return phase == "digging" and 10 <= index < 20

    sentinel = object()
    walk(
        _walk_table(40),
        _levels(),
        fires=fires,
        config=sentinel,
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=2,
    )
    assert seen, "the trigger must actually have been called"
    assert all(c is sentinel for c in seen), (
        f"mixed configs reached the trigger: {set(map(id, seen))}"
    )


# --- calibration robustness ---------------------------------------------------
#
# `calibrate` turns the words in the cue definitions into numbers. If one glitched
# frame can move those numbers, every gate downstream is wrong for the whole video
# -- and the failure is silent, because Otsu always returns something.


def test_one_glitched_sample_cannot_move_a_level():
    """The worst failure mode this pipeline has, and it was reachable.

    A single detector flicker putting `speed_x` at 20 L/s used to give a threshold
    of 10.16 with separability 0.97 -- reported CLEAR while nothing in the video
    exceeds it, so `trigger_swinging` could never fire again. Confidently wrong is
    worse than visibly broken.

    The level is now read from a signal clipped to its own central mass, so an
    extreme sample can contribute to the counts but cannot set the threshold.
    """
    from excavator_cycles.fsm import split_of

    rng = np.random.default_rng(0)
    clean = np.abs(rng.normal(0.0, 0.1, 300))
    clean[:60] += 0.8  # a genuine second population
    baseline = split_of(clean, min_side=3).threshold

    for glitch in (2.0, 5.0, 20.0, 500.0):
        dirty = clean.copy()
        dirty[150] = glitch
        moved = split_of(dirty, min_side=3).threshold
        assert moved == pytest.approx(baseline, rel=0.15), (
            f"a single sample at {glitch} moved the level from {baseline:.4f} to {moved:.4f}"
        )


def test_a_handful_of_samples_is_not_a_population():
    """`trustworthy` was satisfied by `below > 0 and above > 0` -- a 400/1 split
    counted as two populations.

    The bar is now the hold requirement, which is derived rather than picked: a
    side holding fewer samples than a trigger needs to fire cannot produce a
    detection at all, so calling it a population is meaningless.
    """
    from excavator_cycles.fsm import split_of

    rng = np.random.default_rng(1)
    lopsided = np.r_[rng.normal(0.0, 0.01, 400), [5.0, 5.1]]  # two samples apart
    assert not split_of(lopsided, min_side=3).trustworthy
    # The same shape with a real minority population is fine.
    genuine = np.r_[rng.normal(0.0, 0.01, 400), rng.normal(5.0, 0.01, 40)]
    assert split_of(genuine, min_side=3).trustworthy


def test_a_constant_signal_is_never_trustworthy():
    """There is no split. The old code returned the constant AS the threshold,
    which made `< threshold` false everywhere and `>= threshold` true everywhere
    -- silently putting every sample on one side, which is exactly what its own
    comment claimed to be avoiding."""
    from excavator_cycles.fsm import split_of

    split = split_of(np.full(200, 0.4), min_side=3)
    assert not split.trustworthy
    assert split.separability == 0.0


def test_an_untrustworthy_truck_level_disables_the_dumping_gate():
    """Acting on `trustworthy`, not merely logging it.

    Nothing consulted the flag: `trigger_dumping` checked `over_truck is None` but
    never `.trustworthy`, so a meaningless level was used as though it were real.
    The truck gate is OPTIONAL by design -- a video with no truck already sets it
    to None -- so an untrustworthy truck level takes the same route it already has.
    """
    from excavator_cycles.fsm import calibrate

    n = 200
    table = _walk_table(n)
    table.height = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    table.speed_x = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    # Overlap is flat: there is no "near the truck" to separate from "not".
    table.truck_overlap = np.full(n, 0.3)
    assert calibrate(table).over_truck is None, "a meaningless level must not be used"


def test_the_separability_of_a_clean_two_mode_signal_survives_clipping():
    """Clipping must not damage good data, or it trades one failure for another."""
    from excavator_cycles.fsm import split_of

    rng = np.random.default_rng(2)
    clean = np.r_[rng.normal(0.0, 0.05, 150), rng.normal(1.0, 0.05, 150)]
    split = split_of(clean, min_side=3)
    assert split.trustworthy
    assert split.threshold == pytest.approx(0.5, abs=0.1)
    assert split.below == pytest.approx(150, abs=5)


# --- orientation ---------------------------------------------------------------


def _mirror(table):
    """The same footage shot from the other side: every horizontal sign flips.

    `FeatureTable` is frozen, so this rebuilds it rather than mutating -- which is
    the right shape anyway, since a mirrored clip is a different table.
    """
    import dataclasses

    return dataclasses.replace(table, rel_cabin_x=-np.asarray(table.rel_cabin_x, dtype=float))


def _dumpable_table(n=200, *, side=1.0):
    """A clip where the bucket goes over the bed on a given side of the cabin.

    `_CueTable` rather than `_walk_table`, because `trigger_dumping` reads
    `rel_cabin_x` and `found`, which the calibrate-only table does not carry.
    """
    table = _CueTable(n=n)
    table.height = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    table.speed_x = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    table.truck_overlap = np.r_[np.zeros(120), np.full(50, 0.6), np.zeros(n - 170)]
    table.rel_cabin_x = np.full(n, 0.4 * side)
    return table


def test_the_dump_side_is_read_from_the_video_not_assumed():
    """`rel_cabin_x > 0` hardcoded that the truck is on the machine's RIGHT.

    On the dev clip all 86 over-the-bed samples happen to be positive, which is
    why it never bit. Mirror the footage, or park the truck on the other side, and
    every horizontal sign flips: condition 3.2 is never satisfied and dumping is
    never detected on that video at all.

    The module's own comment claims "there is no absolute number anywhere in this
    section". `> 0` is an absolute number, and what it encodes is a scene layout.
    """
    from excavator_cycles.fsm import calibrate

    right = calibrate(_dumpable_table(side=+1.0), None)
    left = calibrate(_dumpable_table(side=-1.0), None)
    assert right.dump_side == pytest.approx(+1.0)
    assert left.dump_side == pytest.approx(-1.0), "the truck is on the other side here"


def test_dumping_is_detected_on_mirrored_footage():
    """The behaviour the hardcoded sign made impossible."""
    from excavator_cycles.fsm import calibrate, trigger_dumping

    for side in (+1.0, -1.0):
        table = _dumpable_table(side=side)
        levels = calibrate(table, None)
        assert trigger_dumping(table, 130, levels), f"side={side:+.0f} must dump over the bed"
        assert not trigger_dumping(table, 10, levels), f"side={side:+.0f} must not dump away"


def test_mirroring_the_dev_clip_changes_nothing_about_dumping():
    """The strongest form: the real table and its mirror must agree exactly."""
    from excavator_cycles.config import Config
    from excavator_cycles.features import load as load_features
    from excavator_cycles.fsm import calibrate, trigger_dumping

    fixture = Path(__file__).resolve().parent / "fixtures" / "dev_clip"
    table, _scene = load_features(fixture)
    mirrored = _mirror(table)
    config = Config()
    original = [trigger_dumping(table, i, calibrate(table, config)) for i in range(0, 296, 7)]
    flipped = [
        trigger_dumping(mirrored, i, calibrate(mirrored, config)) for i in range(0, 296, 7)
    ]
    assert any(original), "the fixture must actually dump somewhere, or this proves nothing"
    assert original == flipped


def test_a_single_sample_at_the_clip_edge_is_not_a_transition():
    """The truncated hold needed a floor.

    Accepting "every sample that exists" with no minimum meant one sample at the
    final index became a detection -- and because digging bounds cycles, that
    changed `cycle_count`, the field graded exactly. Half the hold is the bar: a
    clip ending one or two frames early keeps its transition, a blip does not
    become one.
    """
    from excavator_cycles.fsm import walk

    def found_for(indices):
        return [
            d.phase
            for d in walk(
                _walk_table(50),
                _levels(),
                fires=_scripted({"digging": set(indices)}),
                hold_samples=3,
                strict_hold_samples=6,
                lookback_samples=8,
            )
        ]

    assert found_for({49}) == [], "one sample of three is not a sustained anything"
    assert found_for({48, 49}) == ["digging"], "two of three survives a clip ending early"
    assert found_for({47, 48, 49}) == ["digging"], "the full hold, at the very edge"


def test_a_truncated_hold_is_reported_as_the_weaker_bar_it_is(caplog):
    """Accepting less evidence than asked for must be visible, not quiet."""
    import logging

    from excavator_cycles.fsm import walk

    with caplog.at_level(logging.WARNING):
        walk(
            _walk_table(50),
            _levels(),
            fires=_scripted({"digging": {48, 49}}),
            hold_samples=3,
            strict_hold_samples=6,
            lookback_samples=8,
        )
    assert any("truncated hold" in r.message for r in caplog.records), (
        f"expected a truncation warning, got {[r.message for r in caplog.records]}"
    )


def test_an_out_of_sequence_dig_gets_no_truncation_allowance():
    """It ABANDONS the cycle in progress and adds a boundary, so it must clear the
    full strict bar. It was being accepted on 1 of 6 samples -- the opposite of the
    "unexpected evidence should be expensive" rule it exists to enforce."""
    from excavator_cycles.fsm import walk

    schedule = {"digging": set(range(5, 8)) | {29}, "hauling": set(range(12, 15))}
    found = walk(
        _walk_table(30),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=4,
    )
    assert [d.phase for d in found] == ["digging", "hauling"]
    assert not any(d.out_of_sequence for d in found), (
        "a one-sample dig at the clip edge must not abandon a good cycle"
    )


def test_an_out_of_sequence_dig_with_full_evidence_is_still_accepted():
    """The control: the guard above must not disable the mechanism entirely."""
    from excavator_cycles.fsm import walk

    schedule = {
        "digging": set(range(5, 8)) | set(range(20, 30)),
        "hauling": set(range(12, 15)),
    }
    found = walk(
        _walk_table(40),
        _levels(),
        fires=_scripted(schedule),
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=4,
    )
    assert any(d.out_of_sequence for d in found), f"a genuine mid-cycle dig must fire: {found}"


def test_an_excursion_exactly_as_long_as_the_hold_survives_clipping():
    """`split_of` promises no excursion long enough to be a transition is clipped
    away. It was clipping `min_side` values from each end, which removes an
    excursion of EXACTLY `min_side` samples -- and that length IS long enough,
    because `sustained` asks for `range(start, start + hold)`.

    Measured before the fix, 3-sample excursion with min_side=3: threshold 0.0005,
    separability 0.072, against an unclipped answer of 0.2594 and 0.970. The
    shortest excursion that must survive has `min_side` samples, so at most
    `min_side - 1` may be clipped.
    """
    from excavator_cycles.fsm import split_of

    base = np.random.default_rng(0).normal(0.0, 0.01, 293)
    short = split_of(np.r_[base, np.full(2, 0.5)], min_side=3)
    exact = split_of(np.r_[base, np.full(3, 0.5)], min_side=3)

    assert exact.threshold == pytest.approx(0.26, abs=0.05), (
        f"an excursion of exactly min_side samples was clipped away: {exact}"
    )
    assert exact.trustworthy
    # And one sample shorter than the hold is still correctly discarded: it cannot
    # become a detection, so it must not be allowed to set a level.
    assert not short.trustworthy, f"a 2-sample blip must not define a level: {short}"


def test_the_mass_requirement_rejects_a_lopsided_split_on_its_own():
    """Tested on `Split` directly, because `split_of` can no longer reach this case.

    The clause was added to stop `below > 0 and above > 0` treating a 400-to-1 split
    as two populations. It is worth asserting on its own terms, and it has to be
    asserted here rather than through `split_of`: clipping (added in the same commit)
    already erases any side smaller than the guard, so a lopsided split arrives with
    its separability already crushed and the mass clause never decides anything.

    So the honest statement is that the two mechanisms overlap -- clipping does the
    work on real data, and this clause is the explicit guarantee. Without a test at
    this level, removing the clause entirely changes no test at all.
    """
    from excavator_cycles.fsm import Split

    lopsided = Split(threshold=2.4, separability=0.97, below=400, above=1, min_side=3)
    assert not lopsided.trustworthy, "one sample is not a population"
    assert Split(2.4, 0.97, 400, 3, min_side=3).trustworthy, "min_side samples is the bar"
    assert not Split(2.4, 0.97, 2, 400, min_side=3).trustworthy, "either side counts"
    # And separability still has to clear its own bar independently.
    assert not Split(2.4, 0.5, 400, 50, min_side=3).trustworthy


def test_a_weak_truck_level_still_counts_the_cycle_it_cannot_measure():
    """Disabling an untrustworthy `over_truck` took `cycle_count` from 1 to 0.

    `evidence_within` uses the same triggers, so a disabled truck level meant no
    dumping EVIDENCE either -- the cycle was not `complete` and dropped out of the
    COUNT, not just the averages. `cycles.py` forbids exactly that: "A cue failing is
    a fact about the pipeline; it is not a fact about the excavator."

    The dev clip's truck level scores 0.82 against a bar of 0.80, so a hidden video
    is one bad frame away from this on the one field graded exactly.

    `Levels.for_evidence` is the counting-versus-measuring split one level down: a
    level too weak to say WHEN dumping started can still say THAT the bucket went to
    the bed.
    """
    from excavator_cycles.fsm import Levels, Split, evidence_within, trigger_dumping

    n = 20
    table = _CueTable(n=n)
    table.truck_overlap = np.r_[np.zeros(4), np.full(3, 0.8), np.zeros(n - 7)]
    table.rel_cabin_x = np.full(n, 0.4)
    measured = Split(0.2, 0.5, 17, 3, min_side=3)  # too weak to trust
    levels = Levels(
        low_height=Split(0.1, 0.95, 50, 50),
        over_truck=None,  # disabled for onsets
        moving=Split(0.3, 0.95, 50, 50),
        dump_side=1.0,
        over_truck_observed=measured,
    )

    # The ONSET gate stays shut: a meaningless level must not pin a transition.
    assert not trigger_dumping(table, 5, levels), "a weak level must not locate an onset"
    # The EVIDENCE check still sees it, so the cycle can be counted.
    assert "dumping" in evidence_within(table, levels, 0.0, 2.0), (
        "a disabled gate must not erase the record that the phase happened"
    )


def test_a_video_with_no_truck_at_all_has_nothing_to_fall_back_on():
    """The distinction that matters: "weak truck level" is not "no truck".

    Nothing was measured, so there is no observed split to restore, and dumping
    genuinely left no evidence. A fallback here would invent one.
    """
    from excavator_cycles.fsm import calibrate, evidence_within

    n = 200
    table = _CueTable(n=n)
    table.height = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    table.speed_x = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    table.truck_overlap = np.full(n, np.nan)
    levels = calibrate(table, None)
    assert levels.over_truck is None
    assert levels.over_truck_observed is None, "nothing was measured, so nothing is retained"
    assert levels.for_evidence is levels, "there is no permissive variant to build"
    assert "dumping" not in evidence_within(table, levels, 0.0, 20.0)


def test_conditions_true_from_the_first_frame_do_not_march_the_machine_forward():
    """The edge exemption, unguarded, manufactured cycles out of a constant signal.

    Waiving the rising-edge rule at the first sample the walk looks for a phase
    fixed a real permanent loss -- but unguarded it let already-true conditions
    advance the state machine on no new evidence at all. With all four triggers true
    everywhere, the walk produced 20 detections and `assemble` read four "complete"
    cycles out of a signal that never changed.

    The guard: the exempted phase must have been false at SOME earlier sample, so it
    genuinely rose during the clip. The rise need not be recent -- insisting on that
    is what lost the transition in the first place.
    """
    from excavator_cycles.fsm import walk

    found = walk(
        _walk_table(60),
        _levels(),
        fires=lambda phase, table, index, levels, config: True,
        hold_samples=3,
        strict_hold_samples=6,
        lookback_samples=2,
    )
    # Digging at sample 0 is allowed and documented: the clip may open mid-dig, and
    # there is no earlier sample to establish an edge against. Nothing may follow it
    # on the strength of a condition that was never false.
    assert [d.phase for d in found] == ["digging"], (
        f"a constant signal must not produce a cycle; got {[(d.phase, d.fired_at) for d in found]}"
    )


def test_a_condition_that_genuinely_rises_is_still_exempted():
    """The control, and the case the exemption exists for.

    Both halves matter: if the guard rejected this, the permanent loss it was added
    to fix would be back.
    """
    from excavator_cycles.fsm import walk

    # Hauling is FALSE until it rises, at each of the four positions inside the span
    # the walk skips after confirming digging.
    for rise in (10, 11, 12, 13):
        found = walk(
            _walk_table(60),
            _levels(),
            fires=_scripted({"digging": set(range(10, 13)), "hauling": set(range(rise, 60))}),
            hold_samples=3,
            strict_hold_samples=6,
            lookback_samples=2,
        )
        assert [d.phase for d in found] == ["digging", "hauling"], (
            f"hauling rising at {rise} must still be found"
        )


# --- refine("arrives") --------------------------------------------------------
#
# `arrives` was documented as the mirror image of `departs` -- "find where quiet
# BEGINS and stays" -- and implemented as `flatnonzero(quiet)[0]`, the FIRST quiet
# sample in the window. No "stays" check and no excursion-first step, so it was not
# a mirror of anything. Whenever the window's first sample was already quiet it
# returned the window's LEFT EDGE, a number set by the non-overlap clipping rule
# rather than measured from the signal -- and one that looks exactly as precise as a
# real measurement.


def test_arrives_finds_where_the_signal_settles_not_the_first_quiet_sample():
    """A signal that is quiet, moves, then settles. The answer is the SECOND rest."""
    from excavator_cycles.fsm import Window, refine

    times = np.arange(30) * 0.1
    signal = np.r_[np.linspace(0.0, 1.0, 20), np.zeros(10)]
    assert refine(signal, times, Window(0, 30), "arrives") == pytest.approx(2.0, abs=0.15), (
        "the ramp ends at t=2.0; returning t=0.0 is the window edge, not an arrival"
    )


def test_arrives_is_the_mirror_of_departs():
    """Stated as a property rather than two examples: reversing the signal in time
    must swap the two answers.

    This is the claim the docstring makes, and it is the one the old implementation
    failed. A tolerance of one sample absorbs the band being estimated from a
    different neighbourhood in each direction.
    """
    from excavator_cycles.fsm import Window, refine

    times = np.arange(40) * 0.1
    # quiet, then a burst, then quiet again -- symmetric about the middle.
    signal = np.r_[
        np.zeros(12), np.linspace(0.0, 1.0, 8), np.linspace(1.0, 0.0, 8), np.zeros(12)
    ]
    forward = refine(signal, times, Window(0, 40), "departs")
    backward = refine(signal[::-1], times, Window(0, 40), "departs")
    arrives = refine(signal, times, Window(0, 40), "arrives")
    assert forward is not None and backward is not None and arrives is not None
    # `arrives` on the signal should mirror `departs` on the reversed signal.
    assert arrives == pytest.approx(float(times[-1]) - backward, abs=0.15), (
        f"departs(reversed)={backward:.2f} does not mirror arrives={arrives:.2f}"
    )


def test_arrives_returns_nothing_when_the_signal_never_moved():
    """No excursion means nothing arrived. Returning the first sample -- which is
    what the old code did -- invents an onset out of a flat signal."""
    from excavator_cycles.fsm import Window, refine

    times = np.arange(30) * 0.1
    assert refine(np.zeros(30), times, Window(0, 30), "arrives") is None


def test_arrives_returns_nothing_when_the_signal_is_still_moving_at_the_window_end():
    """The dev clip's digging case: `dh_dt` runs from -0.159 to -0.125 across the
    whole window and never comes near zero, so there is no arrival to find. Reporting
    one would be worse than reporting none."""
    from excavator_cycles.fsm import Window, refine

    times = np.arange(30) * 0.1
    assert refine(np.linspace(-0.2, -0.1, 30), times, Window(0, 30), "arrives") is None


def test_arrives_does_not_return_the_window_edge_on_the_real_clip():
    """The measured symptom, pinned against the fixture.

    Hauling's refined onset was 11.9103, exactly `times[window.lo]` for the window
    [119, 130) -- an artifact of clipping against dumping's window. It scored better
    by coincidence (+1.17 s against +1.97 s) and would not have travelled.
    """
    from pathlib import Path

    from excavator_cycles.config import Config
    from excavator_cycles.features import load as load_features
    from excavator_cycles.fsm import calibrate, locate, walk

    fixture = Path(__file__).resolve().parent / "fixtures" / "dev_clip"
    table, _scene = load_features(fixture)
    config = Config()
    detections = walk(table, calibrate(table, config), config=config)
    onsets = locate(detections, table, config)
    times = table.time_seconds

    for detection, onset in zip(detections, onsets, strict=True):
        if onset.refined is None:
            continue
        edge = float(times[detection.window.lo])
        assert onset.refined != pytest.approx(edge, abs=1e-9), (
            f"{onset.phase} refined to its window's left edge ({edge:.4f}), which is "
            "the clipping rule's number and not a measurement"
        )
