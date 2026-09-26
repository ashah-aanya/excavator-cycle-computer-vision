"""The state machine: calibration, the walk, and pass 2.

Written before the implementation. Every test here states a property the design
document commits to, so a failure means either the code is wrong or the design
changed -- and the second is worth noticing.
"""

from __future__ import annotations

from itertools import pairwise

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
    The config holds durations; the conversion happens once, at the boundary."""
    from excavator_cycles.config import Config

    fsm = Config.load().fsm
    for field in ("hold_seconds", "strict_hold_seconds", "lookback_seconds"):
        assert hasattr(fsm, field), f"FSMConfig is missing {field}"
        assert isinstance(getattr(fsm, field), float)


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
    phase, when = found[0]
    assert phase == "digging"
    assert when == pytest.approx(4.0, abs=0.3), f"arrival is at 4.0s, got {when}"


def test_locate_drops_what_it_cannot_refine_rather_than_guessing():
    """A window with no event in it yields nothing. A made-up onset would flow
    into a duration and be indistinguishable from a measured one."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import Detection, Window, locate

    table = _CueTable(n=80)
    table.dh_dt = np.full(80, -0.3)  # never arrives at rest
    assert locate([Detection("digging", Window(10, 40), 20)], table, Config.load()) == []


def test_locate_keeps_the_onsets_in_order():
    """Refinement must not be able to reorder what the walk ordered. The windows
    cannot overlap, so this is guaranteed by construction -- pinned because if
    it ever breaks, the symptom is a negative phase duration."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import locate, walk

    n = 200
    table = _CueTable(n=n)
    table.height = np.r_[np.linspace(0.5, 0.0, 30), np.zeros(40), np.linspace(0, 0.6, n - 70)]
    table.dh_dt = np.gradient(table.height, table.time_seconds)
    found = walk(table, _lv(low=0.1, moving=5.0), hold_samples=3, lookback_samples=8)
    times = [when for _phase, when in locate(found, table, Config.load())]
    assert times == sorted(times), times


def test_locate_drops_a_detection_whose_cue_found_nothing():
    """refine() returning None means the window held no transition. The
    detection is dropped rather than given a made-up time."""
    from excavator_cycles.config import Config
    from excavator_cycles.fsm import Detection, Window, locate

    table = _CueTable(n=60)  # every signal is flat: nothing to find
    bogus = [Detection("digging", Window(10, 30), 20)]
    assert locate(bogus, table, Config.load()) == []
