"""The state machine: what the features mean, then when the phases changed.

Design diagram stage 3 (``docs/state-machine-design.html``).

The rule this module is built around
------------------------------------
**A global statistic may say what a word MEANS. It may not say WHEN anything
happened.**

Calibration looks at the whole video and works out that, here, "in the material"
means ``height < 0.094 L``. That is a statement about a distribution and it needs
every sample to be any good. Deciding that seconds 2.8 to 12.6 *were* the digging
phase is a different kind of claim, and it belongs to the sequential walk.

An earlier design blurred the two: it cut the video into dig episodes with a
global Otsu split and treated those episodes as the answer. That made digging the
only transition found globally and non-causally while the other three were found
locally and in order -- an asymmetry with no justification, on the one transition
that most wants to behave like the others, because it is the cycle boundary. It
also degrades badly: a single global split assumes the whole clip is bimodal,
where a local search does not care what happened forty seconds ago.

Why Otsu
--------
Otsu's method takes a histogram and finds the cut that makes the two resulting
halves as internally uniform as possible. On this project's signals that is a
real question with a real answer, because each one genuinely mixes two
activities: on the development clip the bucket's height piles up around -0.11 L
while digging and +0.23 L while carrying, with a sparse valley between.

The value of it is that it is **a threshold with no tuned constant**. Writing
``height < 0.1`` bakes in the camera distance, the machine's size and the depth
of the pile. Shoot the same scene from twice as far and every length halves --
but the histogram still has two humps, and Otsu still finds the valley.

What Otsu does not do is notice when it is wrong. Handed one mode, it returns a
number anyway, and that number means nothing. So every split here carries its
**separability** -- Otsu's own objective, normalised: the share of the total
variance that the split explains. Near 1 is two clean modes; low means the cut is
arbitrary and the gate built on it should not be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .geometry import otsu_threshold
from .logging_setup import get_logger

log = get_logger(__name__)

# A split explaining less of the variance than this is reported untrustworthy.
# Dimensionless and fixed across every video: it is a property of the *shape* of
# a distribution, never of any one clip's scale.
#
# Measured, rather than guessed. Otsu always returns a number, so the question is
# where genuinely-one-population ends and genuinely-two begins:
#
#     single Gaussian  0.641  |  two modes, gap 0.3   0.899
#     exponential      0.653  |  two modes, gap 1.0   0.982
#     uniform          0.750  |  zero-inflated + tail 0.835
#                             |  mostly-still+spikes  0.887
#
# The gap between 0.75 and 0.835 is where the line belongs. Note that two of this
# project's three gates -- truck overlap and |dx/dt| -- are not bimodal in the
# mixture-of-Gaussians sense; they are near-zero most of the time with an
# occasional excursion. That is still a real two-population structure and the cut
# is not arbitrary, but it scores lower than a clean pair of modes, and on the
# development clip both land at 0.81-0.82: above the line, without much room.
MIN_SEPARABILITY = 0.80


@dataclass(frozen=True)
class Split:
    """A level derived from one signal's own distribution, and how much to trust it."""

    threshold: float
    separability: float  # [0, 1]: share of the variance the split explains
    below: int  # samples under the threshold
    above: int

    @property
    def trustworthy(self) -> bool:
        """Whether the distribution actually had two modes to separate.

        A gate built on an untrustworthy split is not wrong so much as
        meaningless: the number exists, it just does not correspond to anything.
        """
        return self.separability >= MIN_SEPARABILITY and self.below > 0 and self.above > 0

    def describe(self) -> str:
        verdict = "clear" if self.trustworthy else "WEAK"
        return (
            f"{self.threshold:+.4f}  separability {self.separability:.2f} "
            f"({verdict})  {self.below} below / {self.above} above"
        )


def split_of(signal: np.ndarray) -> Split:
    """Where this signal's two modes divide, and how cleanly.

    ``separability`` is the between-class variance over the total variance --
    exactly the quantity Otsu maximises, divided by a constant so it lands in
    [0, 1] and can be compared across signals with different units.
    """
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite samples; cannot find a split")

    total = float(finite.var())
    if total <= 0:
        # Every sample identical. There is no split, and saying so beats
        # returning a threshold that would silently put everything on one side.
        return Split(float(finite[0]), 0.0, 0, int(finite.size))

    threshold = otsu_threshold(finite)
    below, above = finite[finite < threshold], finite[finite >= threshold]
    if below.size == 0 or above.size == 0:
        return Split(threshold, 0.0, int(below.size), int(above.size))

    weight = below.size / finite.size
    between = weight * (1 - weight) * (float(below.mean()) - float(above.mean())) ** 2
    return Split(threshold, float(between / total), int(below.size), int(above.size))


@dataclass(frozen=True)
class Levels:
    """What the words in the cue definitions mean, on this video.

    Every field is a level. None is an interval, and none of them says when
    anything happened -- see the module docstring for why that separation is the
    point rather than a detail.
    """

    low_height: Split  # "the bucket is down in the material"
    over_truck: Split | None  # "the bucket is at the bed"; None with no truck
    moving: Split  # "the machine is traversing"

    def report(self) -> str:
        lines = [f"  low height   {self.low_height.describe()}"]
        lines.append(
            f"  over truck   {self.over_truck.describe()}"
            if self.over_truck is not None
            else "  over truck   unavailable (no truck detected)"
        )
        lines.append(f"  moving       {self.moving.describe()}")
        return "\n".join(lines)


def calibrate(table) -> Levels:
    """Work out what "low", "over the truck" and "moving" mean on this video.

    One pass over the whole feature table. Nothing here looks at time.
    """
    low_height = split_of(table.height)
    moving = split_of(table.speed_x)

    overlap = np.asarray(table.truck_overlap, dtype=float)
    if not np.isfinite(overlap).any():
        # A legitimate video, not an error: nothing that looked like a truck was
        # ever cleanly detected, so the dumping gate simply has no evidence.
        log.info("no truck overlap in this run; the dumping location gate is unavailable")
        over_truck = None
    else:
        over_truck = split_of(overlap)

    levels = Levels(low_height=low_height, over_truck=over_truck, moving=moving)
    for name, split in (
        ("low height", low_height),
        ("over truck", over_truck),
        ("moving", moving),
    ):
        if split is not None and not split.trustworthy:
            log.warning(
                "the %s level rests on a weak split (%s). Otsu returns a number "
                "whether or not the data has two modes; this one probably does not.",
                name,
                split.describe(),
            )
    return levels


# The cycle, in order. Fixed by the task definition, not by this video: digging,
# hauling, dumping, swinging, and round again.
PHASES = ("digging", "hauling", "dumping", "swinging")


@dataclass
class MachineState:
    """Where the walk is, and what it has collected.

    Starts in ``swinging`` so the first thing it looks for is ``digging``.
    Starting anywhere else would mean guessing what the machine was doing before
    the clip began, and it costs nothing: a cycle runs from one digging onset to
    the next, so a leading partial is discarded either way.
    """

    curr_stage: str = "swinging"
    since: float | None = None

    # Onsets located in the cycle currently being built.
    pending: dict[str, float] = field(default_factory=dict)
    # Phases that left evidence of having happened, whether or not an onset was
    # located for them. This is the weak second check, and it is what separates
    # "the cue failed" from "no cycle happened" when a cycle is closed.
    occurred: set[str] = field(default_factory=set)

    @property
    def looking_for(self) -> str:
        """The one transition that can legally come next."""
        return PHASES[(PHASES.index(self.curr_stage) + 1) % len(PHASES)]

    def elapsed(self, now: float) -> float | None:
        """How long we have been in this phase. ``None`` before the first onset."""
        return None if self.since is None else now - self.since

    def advance(self, phase: str, when: float) -> None:
        """Accept a located onset and move into that phase.

        Refuses anything out of order. Ordering is meant to be impossible to
        violate rather than checked after the fact -- a transition accepted out
        of sequence would surface later as a nonsense duration rather than as an
        error, which is far harder to notice.
        """
        if phase != self.looking_for:
            raise ValueError(
                f"looking for {self.looking_for}, not {phase} (currently in {self.curr_stage})"
            )
        if self.since is not None and when < self.since:
            raise ValueError(f"onset at {when:.3f}s runs backwards from {self.since:.3f}s")
        self.pending[phase] = when
        self.occurred.add(phase)
        self.curr_stage = phase
        self.since = when


def samples_for(seconds: float, times: np.ndarray) -> int:
    """How many samples span ``seconds``, from the observed spacing.

    The single place a duration becomes a count. The median spacing is used
    rather than the mean so one dropped frame cannot stretch every window, and
    the result is never below 1 -- a parameter shorter than the sample interval
    means "as soon as possible", not "never".
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return 1
    spacing = float(np.median(np.diff(times)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"timestamps are not increasing (median spacing {spacing})")
    return max(1, round(seconds / spacing))


@dataclass(frozen=True)
class Window:
    """A half-open span of SAMPLE INDICES, not seconds.

    Seconds and indices are deliberately different types. Config is always in
    seconds; the conversion happens once, at the boundary. Mixing them is how a
    window ends up right on one video and wrong on another.
    """

    lo: int
    hi: int

    def __len__(self) -> int:
        return max(0, self.hi - self.lo)


@dataclass(frozen=True)
class Detection:
    """A transition the walk believes happened, and roughly where.

    Not a time yet. Pass 2 turns the window into an instant; until then this
    says only "the transition into `phase` happened somewhere in here".
    """

    phase: str
    window: Window
    fired_at: int  # the sample where the evidence STARTED, not where it was confirmed
    out_of_sequence: bool = False  # a dig that interrupted another phase


def walk(
    table,
    levels: Levels,
    fires=None,
    config=None,
    hold_samples: int | None = None,
    strict_hold_samples: int | None = None,
    lookback_samples: int | None = None,
) -> list[Detection]:
    """Slide over the samples, watching for the one transition that can come next.

    Args:
        fires: ``(phase, table, index, levels, config) -> bool`` -- does the
            trigger for ``phase`` hold at this sample? Injected so the walk's
            sequencing can be tested without a view on whether any cue is good.
        hold_samples: how long evidence must persist before it is believed. A
            trigger is not a transition; one noisy sample must not advance the
            state.
        strict_hold_samples: the bar for an out-of-sequence digging trigger.
            Higher by default, because expected evidence is cheap and unexpected
            evidence should be expensive -- a spurious dig mid-haul would
            silently truncate a good cycle. Defaults to twice ``hold_samples``.
        lookback_samples: how far the coarse window reaches back before the
            trigger. An onset is where a signal LEFT rest, and that is found by
            walking backward from the excursion, so a window starting at the
            trigger would exclude the thing it is looking for.
        config: supplies all three as DURATIONS, which is how callers should
            pass them -- the sample counts above exist for tests that want to
            pin an exact number of samples, and override the config when given.

    Returns the detections in the order they were found. Multi-cycle is not
    special-cased: the loop simply keeps going.
    """
    if fires is None:
        fires = default_fires

    times = np.asarray(table.time_seconds, dtype=float)
    if config is not None:
        hold_samples = hold_samples or samples_for(config.fsm.hold_seconds, times)
        strict_hold_samples = strict_hold_samples or samples_for(
            config.fsm.strict_hold_seconds, times
        )
        lookback_samples = lookback_samples or samples_for(config.fsm.lookback_seconds, times)
    hold_samples = hold_samples if hold_samples is not None else 3
    lookback_samples = lookback_samples if lookback_samples is not None else 8
    strict = strict_hold_samples if strict_hold_samples is not None else hold_samples * 2

    count = len(times)
    state = MachineState()
    found: list[Detection] = []
    index = 0
    # Where the previous transition was detected. Every window is clipped to it,
    # so consecutive windows cannot overlap.
    previous = None

    def sustained(phase: str, start: int, needed: int) -> bool:
        """Did the trigger RISE here and then hold for `needed` samples?

        A rising edge, not a level. The triggers are written as conditions --
        "the bucket is down and still" -- and such a condition is true for the
        whole of the phase it describes, not only at its start. Accepting a
        level would make the always-on digging check fire on every sample of a
        dig it has already recorded: on the development video that produced
        sixteen digging detections inside one digging phase.

        The edge is the whole difference between "digging is happening" and
        "digging just started", and only the second is a transition.
        """
        if start + needed > count:
            return False
        if start > 0 and fires(phase, table, start - 1, levels, config):
            return False  # already true before this sample: not an edge
        return all(fires(phase, table, i, levels, None) for i in range(start, start + needed))

    while index < count:
        target = state.looking_for

        # The transition we are expecting.
        if sustained(target, index, hold_samples):
            detected = _detect(
                state, target, index, hold_samples, lookback_samples, times, previous
            )
            found.append(detected)
            previous = detected.window.hi
            index += hold_samples
            continue

        # Digging, always, at a higher bar -- it is the only transition that
        # re-establishes where we are.
        #
        # No guard against digging interrupting itself is needed: `sustained`
        # requires a rising edge, and the condition that started a dig stays true
        # for as long as that dig lasts, so it cannot rise again until the bucket
        # has actually come back up. A `curr_stage != "digging"` guard here would
        # be redundant AND wrong -- it would also block a genuine second dig.
        if target != "digging" and sustained("digging", index, strict):
            log.info(
                "digging at sample %d interrupted %s; the cycle being built is abandoned",
                index,
                state.curr_stage,
            )
            state.curr_stage = "swinging"  # so digging is legal again
            state.pending.clear()
            state.occurred.clear()
            state.since = None
            detection = _detect(
                state, "digging", index, strict, lookback_samples, times, previous
            )
            found.append(
                Detection(detection.phase, detection.window, detection.fired_at, True)
            )
            previous = detection.window.hi
            index += strict
            continue

        index += 1

    return found


def _detect(state, phase, index, held, lookback, times, previous=None) -> Detection:
    """Record the transition and move the state into it.

    ``previous`` is where the previous window ENDED, not where it was triggered.
    Clipping to the trigger is not enough: a window extends forward past its own
    trigger by ``held`` samples, so two windows would still overlap by exactly
    that much. On the development video that left hauling and dumping sharing
    three samples even after the clip was added.

    This matters only once pass 2 exists, and then it matters a lot: pass 2
    searches INSIDE a window, so overlapping windows would let it return hauling
    at 13.0 s and dumping at 12.8 s -- a negative phase duration, out of a
    machine whose whole design is that ordering cannot be violated. Ordering is
    enforced on the onset times by `MachineState.advance`, and that guarantee is
    worth nothing if refinement is then free to reorder them.

    No grace margin. An earlier draft let the window reach a little before the
    previous onset, reasoning that a departure cue has to walk backward to find
    where a signal left rest. That was wrong: this phase's onset must be after
    the previous phase started, which is not a convention but what "phase"
    means. A cue needing to look further back than that is reporting that the
    previous onset was wrong, and the right response is to notice, not to allow
    an impossible answer.
    """
    lo = max(0, index - lookback)
    if previous is not None:
        lo = max(lo, previous)
    window = Window(lo, min(len(times), index + held))
    state.advance(phase, float(times[index]))  # provisional; pass 2 refines it
    return Detection(phase, window, index)


# ---------------------------------------------------------------------------
# The coarse triggers -- pass 1
#
# Each one answers "does this sample look like the next phase is starting?" and
# nothing more. They pick the WINDOW; pass 2 picks the instant. That division is
# why they are allowed to be blunt: a trigger that fires a little early or a
# little wide costs pass 2 a slightly longer search, while a trigger that never
# fires loses the transition entirely. So when in doubt, these err towards
# firing.
#
# Every threshold below is a level from `calibrate()` -- derived from this
# video's own distribution. There is no absolute number anywhere in this section,
# which is what lets the same code work on footage shot from a different distance.
# ---------------------------------------------------------------------------


def _usable(table, index: int, *columns: str) -> bool:
    """Was this sample measured at all?

    A missing sample is not evidence against a transition -- the design treats
    gaps as unknown rather than negative -- but it is certainly not evidence
    *for* one, so nothing may fire on it.
    """
    if not bool(np.asarray(table.found)[index]):
        return False
    return all(np.isfinite(getattr(table, name)[index]) for name in columns)


def trigger_digging(table, index: int, levels: Levels) -> bool:
    """CUE 1  swinging -> digging.

    The spec: *digging begins when the bucket first contacts the material and
    starts scooping.*

    Two conditions, and the second is what stops this firing on every pass over
    the pile:

      1.1  the bucket is DOWN -- `height` below the level that separates "in the
           material" from "carrying" on this video. The diagram's "vertical goes
           to the floor", with the floor derived rather than defined.

      1.2  the machine is NOT traversing -- `|dx/dt|` below the "moving" level.
           Low and moving is the bucket passing through on its way somewhere,
           which the spec calls swinging. Low and still is a scoop. This is the
           diagram's "no horizontal motion".

    Deliberately NOT a level crossing of a calibrated surface height. `height`
    carries a per-video offset (the slew centre moves 0.098 L across a quantile
    sweep, 22% of its whole range), so an absolute threshold on it would not
    travel. Being below a level derived from the same distribution does travel.
    """
    if not _usable(table, index, "height", "speed_x"):
        return False
    return (
        table.height[index] < levels.low_height.threshold
        and table.speed_x[index] < levels.moving.threshold
    )


def trigger_hauling(table, index: int, levels: Levels) -> bool:
    """CUE 2  digging -> hauling.

    The spec: *hauling begins when the entire loaded bucket clears the material
    surface and continues as it moves toward the dumping location.*

      2.1  the bucket is ABOVE the material level, and
      2.2  it is RISING -- `dh_dt > 0`.

    Condition 2.2 is the one that matters. The bucket is above the material for
    most of the cycle, so height alone would fire continuously from the moment it
    lifts until it comes back down. The spec's word is *clears*, which describes
    a direction of travel, not a position.

    `dh_dt` rather than `height` also sidesteps the offset problem: a rate is
    immune to where the origin sits, while a level is not.
    """
    if not _usable(table, index, "height", "dh_dt"):
        return False
    return table.height[index] > levels.low_height.threshold and table.dh_dt[index] > 0


def trigger_dumping(table, index: int, levels: Levels) -> bool:
    """CUE 3  hauling -> dumping.

    The spec: *dumping begins when the bucket reaches the dumping location and
    starts tipping or uncurling to release its load.* And, crucially: *material
    that spills while the loaded bucket is still being lifted or transported
    remains part of hauling.*

      3.1  the bucket is OVER THE BED -- `truck_overlap` above the level that
           separates "near the truck" from "not". The diagram's "closest to the
           truck".

      3.2  the bucket is OUT PAST THE CABIN -- `rel_cabin_x > 0`. The diagram's
           "cabin is to the left of the bucket" and "higher x value than cabin",
           expressed as a sign so it does not depend on which way the machine
           happens to face.

    THE GAP WORTH KNOWING ABOUT. The spec's anti-spillage rule really asks for
    the bucket's own ROTATION -- material falling during transport is still
    hauling, so only the bucket tipping should start dumping. Neither condition
    above measures rotation. The guard here rests entirely on position, and
    whether that is sufficient is genuinely untested.

    `over_truck` is None on a video where no truck was ever cleanly detected.
    That is a legitimate video, and the gate is then UNAVAILABLE rather than
    false -- but with no positional evidence at all this returns False, which
    means dumping will never be detected on such a clip. A known limitation.
    """
    if levels.over_truck is None:
        return False
    if not _usable(table, index, "truck_overlap", "rel_cabin_x"):
        return False
    return (
        table.truck_overlap[index] > levels.over_truck.threshold
        and table.rel_cabin_x[index] > 0
    )


def trigger_swinging(table, index: int, levels: Levels) -> bool:
    """CUE 4  dumping -> swinging.

    The spec: *swinging begins when the excavator starts rotating back the
    emptied bucket toward the digging location.*

      4.1  the machine is TRAVERSING -- `|dx/dt|` above the "moving" level, and
      4.2  the bucket is DESCENDING -- `dh_dt < 0`.

    The diagram calls this "opposite conditions to hauling", and 4.2 is the
    opposite in question. Both phases are traverses; the machine is moving
    sideways in each. What separates them is the vertical: hauling carries a
    loaded bucket UP and out, swinging brings an empty one BACK DOWN.

    Note what is absent. The spec says "rotating back TOWARD the digging
    location", which is a direction. This tests the vertical sign instead, which
    is correlated with it but not the same thing -- a machine that repositioned
    without returning would still satisfy both conditions. Measuring the
    direction properly needs a rotation signal, and this pipeline deliberately
    has none: the two earlier attempts at one both measured the wrong quantity.
    """
    if not _usable(table, index, "speed_x", "dh_dt"):
        return False
    return table.speed_x[index] > levels.moving.threshold and table.dh_dt[index] < 0


TRIGGERS = {
    "digging": trigger_digging,
    "hauling": trigger_hauling,
    "dumping": trigger_dumping,
    "swinging": trigger_swinging,
}


def default_fires(phase: str, table, index: int, levels: Levels, _config=None) -> bool:
    """The trigger the walk uses when none is injected.

    ``_config`` is part of the injection protocol -- a caller supplying its own
    trigger may want it -- and is unused here, hence the underscore.
    """
    return TRIGGERS[phase](table, index, levels)


# ---------------------------------------------------------------------------
# Pass 2 -- exactly when, inside a window pass 1 has already chosen
#
# Pass 1 is allowed to be blunt because pass 2 is precise. The division only
# works one way round, though: refine() searches INSIDE the window it is given
# and cannot reach outside it, so a window that misses the transition cannot be
# rescued here. A refined answer from a wrong window is worse than none, because
# it looks precise.
# ---------------------------------------------------------------------------

Mode = Literal["departs", "arrives", "peak"]


def refine(
    signal: np.ndarray,
    times: np.ndarray,
    window: Window,
    mode: Mode,
    sigma: float = 3.0,
    floor_fraction: float = 0.02,
) -> float | None:
    """The instant a transition happened, inside ``window``.

    Three shapes of onset, because the four cues are not all the same kind:

    ``departs``  the signal leaves rest. Found by locating the excursion and
                 walking BACKWARD to where it started -- you cannot detect a
                 departure going forwards, because a signal at rest looks
                 identical to one about to move.
    ``arrives``  the signal settles into rest. The mirror image, walking forward.
    ``peak``     a turning point, for a cue whose event is an extremum rather
                 than a change of regime (T3's aspect ratio).

    Returns ``None`` when nothing qualifies in the window. That is information
    -- "the cue did not fire here" -- and the caller uses it to decide whether a
    cycle is measurable, so it must not be an exception.

    Every time returned is read from ``times``. Never ``t0 + i * dt``: on a
    variable-rate clip that is wrong, and wrong without any symptom.
    """
    values = np.asarray(signal, dtype=float)
    times = np.asarray(times, dtype=float)
    lo, hi = max(0, window.lo), min(len(values), window.hi)
    if hi - lo < 2:
        return None

    inside = values[lo:hi]
    finite = np.isfinite(inside)
    if finite.sum() < 2:
        return None

    if mode == "peak":
        # argmax over the window. An extremum is unmoved by symmetric smoothing,
        # which is why it can be taken directly rather than reconstructed.
        return float(times[lo + int(np.nanargmax(np.where(finite, inside, -np.inf)))])

    # Rest for a RATE is zero, absolutely -- it does not have to be estimated,
    # only its width does. Estimating the level is what once made the detector
    # settle on the hauling height instead of the dig plateau.
    band = _rest_band(inside[finite], sigma, floor_fraction)
    if band <= 0:
        return None
    quiet = np.abs(inside) <= band

    if mode == "departs":
        # Find the excursion, then walk BACK to where the quiet ended.
        loud = np.flatnonzero(~quiet & finite)
        if loud.size == 0:
            return None
        cursor = int(loud[0])
        while cursor > 0 and not quiet[cursor - 1]:
            cursor -= 1
        return float(times[lo + cursor])

    if mode == "arrives":
        # The mirror: find where quiet BEGINS and stays.
        settled = np.flatnonzero(quiet & finite)
        if settled.size == 0:
            return None
        return float(times[lo + int(settled[0])])

    raise ValueError(f"unknown mode {mode!r}")


def _rest_band(values: np.ndarray, sigma: float, floor_fraction: float = 0.02) -> float:
    """How far from zero still counts as "at rest".

    Two traps here, both of which this project has fallen into before.

    **The median difference is not the noise.** A window chosen because it
    contains a transition is, by construction, mostly moving -- so the MEDIAN of
    successive differences measures the motion. On a clean synthetic ramp that
    estimate came out at the ramp's own step size, 0.052, which then swallowed
    the entire excursion and put the onset four samples late. A low quantile
    looks at the quiet part instead.

    **And it can come out at zero.** On a signal that is exactly flat before the
    transition, the low quantile is 0, and a band of zero makes every sample an
    excursion. The floor is a small fraction of what the signal does across the
    window: rest cannot be defined more tightly than that.
    """
    if values.size < 3:
        return 0.0
    steps = np.abs(np.diff(values))
    quiet = float(np.quantile(steps, 0.25)) * 1.4826 / np.sqrt(2)
    return max(sigma * quiet, floor_fraction * float(np.ptp(values)))


# Which signal each transition's onset lives in, and what shape it has.
#
# These are the cues the recovered detector used, and the reason they belong in
# pass 2 as well as pass 1 is that they describe the EVENT rather than the state
# around it. `digging` is the bucket stopping its descent, which is contact;
# `hauling` is the kink where scooping becomes lifting; `swinging` is the machine
# leaving rest; `dumping` is the bucket's silhouette at its most stretched, which
# is the closest thing available to "it has tipped".
REFINEMENTS: dict[str, tuple[str, Mode]] = {
    "digging": ("dh_dt", "arrives"),
    "hauling": ("d2h_dt2", "arrives"),
    "dumping": ("aspect_ratio", "peak"),
    "swinging": ("speed_x", "departs"),
}


def locate(detections: list[Detection], table, config) -> list[tuple[str, float]]:
    """Turn pass 1's windows into instants: the join between the two passes.

    A detection whose cue finds nothing in its window is DROPPED rather than
    given the trigger time as a fallback. A made-up onset would flow into a
    duration and be indistinguishable from a measured one; a missing onset is
    visible, and `Cycle.reason` can say so.

    Ordering needs no enforcement here -- pass 1's windows cannot overlap, so a
    refined time cannot cross its neighbour. That is checked by a test rather
    than asserted at runtime, because if it ever breaks the symptom is a
    negative phase duration rather than an exception.
    """
    times = np.asarray(table.time_seconds, dtype=float)
    out: list[tuple[str, float]] = []
    for detection in detections:
        column, mode = REFINEMENTS[detection.phase]
        when = refine(
            getattr(table, column),
            times,
            detection.window,
            mode,
            sigma=config.fsm.rest_sigma,
            floor_fraction=config.fsm.rest_floor_fraction,
        )
        if when is None:
            log.info(
                "%s at sample %d: the %s cue found no %s in [%d, %d); dropped",
                detection.phase,
                detection.fired_at,
                column,
                mode,
                detection.window.lo,
                detection.window.hi,
            )
            continue
        out.append((detection.phase, when))
    return out
