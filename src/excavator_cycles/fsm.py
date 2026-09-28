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

import math
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Literal

import numpy as np

from .config import FeatureConfig, FSMConfig
from .geometry import otsu_threshold
from .logging_setup import get_logger
from .onsets import derivative
from .shapes import Reading
from .shapes import read as read_shape

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

    min_side: int = 1  # the smallest side that counts as a population; see `split_of`

    @property
    def trustworthy(self) -> bool:
        """Whether the distribution actually had two modes to separate.

        A gate built on an untrustworthy split is not wrong so much as
        meaningless: the number exists, it just does not correspond to anything.

        The mass requirement is not decoration. ``below > 0 and above > 0`` was
        satisfied by a 400-to-1 split, so a single stray sample could constitute a
        whole "population" and a level could sit above 99.99% of the data while
        reporting itself clear.
        """
        return self.separability >= MIN_SEPARABILITY and min(self.below, self.above) >= max(
            1, self.min_side
        )

    def describe(self) -> str:
        verdict = "clear" if self.trustworthy else "WEAK"
        return (
            f"{self.threshold:+.4f}  separability {self.separability:.2f} "
            f"({verdict})  {self.below} below / {self.above} above"
        )


def split_of(signal: np.ndarray, min_side: int = 1) -> Split:
    """Where this signal's two modes divide, and how cleanly.

    ``separability`` is the between-class variance over the total variance --
    exactly the quantity Otsu maximises, divided by a constant so it lands in
    [0, 1] and can be compared across signals with different units.

    **The threshold is read from a CLIPPED copy of the signal.** Otsu maximises a
    variance, so it is pulled by extremes: a single detector flicker putting
    `speed_x` at 20 L/s moved the level to 10.16 -- above every real sample, so
    the swinging gate could never fire again -- and reported separability 0.97,
    i.e. CLEAR. Confidently wrong for a whole video, from one frame. Clipping to
    the signal's own central mass lets an outlier count towards the tallies while
    denying it the power to place the line.

    How much is clipped is derived, not chosen: ``min_side - 1`` samples from each
    end, one short of the hold requirement, and none at all when ``min_side`` is 1.
    No excursion long enough to be a transition is ever clipped away, because an
    excursion shorter than the hold cannot become a detection anyway. At 296
    samples with a 3-sample hold that is 2 samples from each end.

    ``separability`` and the two counts are computed on the UNCLIPPED data, so a
    heavy tail still reads as the poor split it is rather than being tidied away.

    Args:
        min_side: the smallest number of samples on one side that counts as a
            population, and the amount clipped from each end. Pass the walk's hold
            in samples; the default of 1 keeps the old behaviour for callers that
            have no sample rate to hand.
    """
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite samples; cannot find a split")

    total = float(finite.var())
    if total <= 0:
        # Every sample identical. There is genuinely no split. The threshold
        # returned here is the constant itself, which puts every sample on one
        # side -- unavoidable for a float return, and the reason `trustworthy` is
        # False rather than something the caller has to infer from the counts.
        return Split(float(finite[0]), 0.0, 0, int(finite.size), min_side)

    # `min_side - 1`, not `min_side`. Clipping `min_side` values from each end
    # removes an excursion of EXACTLY `min_side` samples -- which is long enough to
    # fire, since `sustained` asks for `range(start, start + hold)` -- and so
    # contradicted this function's own promise that no excursion long enough to be a
    # transition is ever clipped away. Measured with a 3-sample excursion and
    # min_side=3: threshold 0.0005 and separability 0.072, against an unclipped
    # answer of 0.2594 and 0.970. The shortest excursion that must survive has
    # `min_side` samples, so the most that may be clipped is `min_side - 1`.
    #
    # And no floor of 1. `max(1, ...)` forced one sample off each end even at
    # min_side=1, where the rule above allows ZERO -- and a 1-sample excursion is
    # exactly what min_side=1 says can fire. 199 zeros and one 10.0 came back as
    # threshold 0.0, separability 0.0: the excursion erased and the split denied.
    guard = max(0, min(int(min_side) - 1, (finite.size - 1) // 2))
    if guard > 0 and finite.size > 2 * guard:
        keep = np.sort(finite)[guard:-guard]
        threshold = otsu_threshold(np.clip(finite, keep[0], keep[-1]))
    else:
        threshold = otsu_threshold(finite)

    below, above = finite[finite < threshold], finite[finite >= threshold]
    if below.size == 0 or above.size == 0:
        return Split(threshold, 0.0, int(below.size), int(above.size), min_side)

    weight = below.size / finite.size
    between = weight * (1 - weight) * (float(below.mean()) - float(above.mean())) ** 2
    return Split(threshold, float(between / total), int(below.size), int(above.size), min_side)


@dataclass(frozen=True)
class Levels:
    """What the words in the cue definitions mean, on this video.

    Every field is a level. None is an interval, and none of them says when
    anything happened -- see the module docstring for why that separation is the
    point rather than a detail.
    """

    low_height: Split  # "the bucket is down in the material"
    over_truck: Split | None  # "the bucket is at the bed"; None when unusable
    moving: Split  # "the machine is traversing"
    # Which side of the cabin the truck is on: +1 right, -1 left. A level like the
    # others -- read off this video rather than assumed. See `calibrate`.
    dump_side: float = 1.0
    # The truck level AS MEASURED, retained even when it is too weak to use for an
    # onset. See `for_evidence` below for why both are kept.
    over_truck_observed: Split | None = None
    # NOT a level: the local shape of each signal the onset cues read, at every
    # sample (`shapes.py`). Carried here because every onset cue needs it and it is
    # computed once, in `calibrate`. The only whole-video statistic inside it is
    # the signal's spread, which says what "flat" MEANS -- a level, as this class
    # requires. Where each shape occurs is read from a few seconds around each
    # sample, never from the whole clip. None for a table that lacks the columns,
    # such as the scripted tables the sequencing tests build.
    readings: dict[str, Reading] | None = None

    @property
    def for_evidence(self) -> Levels:
        """The same levels, with a weak truck level allowed back in.

        The counting-versus-measuring split, one level down. A truck level that
        cannot be trusted to say WHEN dumping started can still say THAT the bucket
        went to the bed at some point, and those are different questions -- the
        second is the weak check `cycles.py` counts cycles on.

        Without this, disabling an untrustworthy `over_truck` took `cycle_count` from
        1 to 0 on the dev clip, because `evidence_within` uses the same triggers: no
        dumping evidence meant the cycle was not `complete` and dropped out of the
        COUNT, not just the averages. That is what `cycles.py` forbids in as many
        words -- "A cue failing is a fact about the pipeline; it is not a fact about
        the excavator." The dev clip's truck level scores 0.82 against a bar of 0.80,
        so a hidden video is one bad frame from that outcome on the one field graded
        exactly.
        """
        if self.over_truck is not None or self.over_truck_observed is None:
            return self
        return replace(self, over_truck=self.over_truck_observed)

    def report(self) -> str:
        side = "right" if self.dump_side >= 0 else "left"
        lines = [f"  dump side    the truck is on the machine's {side}"]
        lines.append(f"  low height   {self.low_height.describe()}")
        lines.append(
            f"  over truck   {self.over_truck.describe()}"
            if self.over_truck is not None
            else "  over truck   unavailable (no truck detected)"
        )
        lines.append(f"  moving       {self.moving.describe()}")
        return "\n".join(lines)


def calibrate(table, config=None) -> Levels:
    """Work out what "low", "over the truck" and "moving" mean on this video.

    One pass over the whole feature table. Nothing here looks at time -- except to
    convert the hold duration into a sample count, which decides both how much of
    each signal is clipped before the threshold is read and how many samples one
    side must hold to count as a population. See `split_of`.

    **What happens to a level that cannot be trusted.** Previously: a WARNING was
    logged and the number was used anyway, which is the one outcome worse than
    either alternative -- the estimator knew it had failed and the answer came out
    confident. Now it depends on whether the design can do without the gate:

    * ``over_truck`` is OPTIONAL. A video with no truck already sets it to None and
      `in_dumping` already handles that, so an untrustworthy truck level takes the
      same route. That affects only the dumping STATE; the dumping ONSET cue
      (`trigger_dumping`) reads the height's shape and `dump_side`, not this level,
      so dumping is still detected.
    * ``low_height`` and ``moving`` are REQUIRED -- without them nothing can fire at
      all. Disabling them would turn a degraded answer into no answer, and the task
      asks for an answer, so these are kept and the warning says plainly that the
      result is unreliable. The clipping in `split_of` is what makes that tolerable:
      it removes the catastrophic case where one frame moved a level clean outside
      the data.
    """
    hold = FSMConfig().hold_seconds if config is None else config.fsm.hold_seconds
    min_side = samples_for(hold, np.asarray(table.time_seconds, dtype=float))

    low_height = split_of(table.height, min_side)
    moving = split_of(table.speed_x, min_side)

    overlap = np.asarray(table.truck_overlap, dtype=float)
    if not np.isfinite(overlap).any():
        # A legitimate video, not an error: nothing that looked like a truck was
        # ever cleanly detected, so the dumping gate simply has no evidence.
        log.info("no truck overlap in this run; the dumping location gate is unavailable")
        # Nothing was measured, so there is nothing to fall back on for evidence
        # either. This is genuinely "no truck", not "a weak truck level".
        over_truck = over_truck_observed = None
    else:
        over_truck = split_of(overlap, min_side)
        over_truck_observed = over_truck
        if not over_truck.trustworthy:
            log.warning(
                "the over-truck level is meaningless (%s); the dumping STATE check is "
                "disabled for this video. The dumping onset cue does not use this level, "
                "so dumping is still detected.",
                over_truck.describe(),
            )
            over_truck = None

    # WHICH SIDE the truck is on, derived rather than assumed. `rel_cabin_x > 0`
    # hardcoded "the truck is on the machine's right": true of the dev clip, where
    # all 86 over-the-bed samples are positive, and false the moment the footage is
    # mirrored or the truck is parked on the other side -- condition 3.2 would then
    # never be satisfied and dumping would never be detected on that video at all.
    #
    # The question the data can answer is: at the samples where the bucket is most
    # over the bed, which side of the cabin is it on? The median sign of those is
    # the dump side. A median rather than a mean so a few bad boxes cannot flip it.
    #
    # Read from the level AS MEASURED, not only a trusted one. `for_evidence` puts a
    # weak truck level back to COUNT cycles, and `trigger_dumping` multiplies by
    # this sign there too -- so deriving it only when the level was trusted left
    # the default +1 in force for exactly the weak case, and a mirrored clip with a
    # weak truck level lost its dumping evidence and dropped `cycle_count` from 1
    # to 0. "Which side is the bed on" is a THAT question, like the one
    # `for_evidence` exists for; it does not need a level fit to say WHEN.
    dump_side = 1.0
    if over_truck_observed is not None:
        overlap_finite = np.isfinite(overlap)
        at_bed = overlap_finite & (overlap > over_truck_observed.threshold)
        rel = np.asarray(table.rel_cabin_x, dtype=float)
        usable = rel[at_bed & np.isfinite(rel)]
        if usable.size:
            middle = float(np.median(usable))
            # A median of exactly 0 means the bucket straddles the cabin at the bed
            # and the signal cannot say. Keeping +1 there is arbitrary, so say so.
            if middle == 0.0:
                log.warning(
                    "the bucket straddles the cabin over the bed; the dump side "
                    "cannot be determined and is assumed to be the machine's right"
                )
            else:
                dump_side = 1.0 if middle > 0 else -1.0
        else:
            log.warning("no usable rel_cabin_x over the bed; assuming the dump side is right")

    for name, split in (("low height", low_height), ("moving", moving)):
        if not split.trustworthy:
            log.warning(
                "the %s level rests on a weak split (%s). Otsu returns a number "
                "whether or not the data has two modes; this one probably does not. "
                "This level is REQUIRED, so it is being used anyway and the answer "
                "for this video should be treated as unreliable.",
                name,
                split.describe(),
            )
    return Levels(
        low_height=low_height,
        over_truck=over_truck,
        moving=moving,
        dump_side=dump_side,
        over_truck_observed=over_truck_observed,
        readings=read_signals(table, config, dump_side),
    )


# The signals the onset cues read shapes from. None is stored in `features.npz`
# as such; they are derived here from `dx_dt`, `bucket_y` and `height`, with the
# same Savitzky-Golay derivative and window that built the stored rates.
#
# `d2x_toward_truck` is the horizontal acceleration signed so that + is TOWARD the
# truck. Raw d2x/dt2 is not orientation independent: mirror the footage and every
# horizontal sign flips, turning the return swing's peak into a dip. Multiplying by
# `dump_side` -- which side the truck is on, read from this video -- makes the cue
# the same event whichever way the camera faces.


def read_signals(table, config=None, dump_side: float = 1.0) -> dict[str, Reading] | None:
    """The shape of every cue signal at every sample, or None if the table cannot say."""
    if not all(hasattr(table, c) for c in ("time_seconds", "dx_dt", "bucket_y", "height")):
        return None
    fsm = FSMConfig() if config is None else config.fsm
    window = (FeatureConfig() if config is None else config.features).derivative_window_seconds
    times = np.asarray(table.time_seconds, dtype=float)
    dx = np.asarray(table.dx_dt, dtype=float)
    dy = derivative(np.asarray(table.bucket_y, dtype=float), times, window_seconds=window)
    signals = {
        "speed_2d": np.hypot(dx, dy),
        "height": np.asarray(table.height, dtype=float),
        "d2x_toward_truck": derivative(dx, times, window_seconds=window) * dump_side,
    }
    return {
        name: read_shape(
            values,
            times,
            side=fsm.shape_side_seconds,
            flat=fsm.shape_flat_fraction,
            steeper=fsm.shape_steeper,
            min_side=fsm.shape_min_side_seconds,
        )
        for name, values in signals.items()
    }


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

    # NOTE: this class deliberately carries no per-cycle bookkeeping. It used to
    # hold `pending` (onsets located so far) and `occurred` ("the weak second
    # check... what separates 'the cue failed' from 'no cycle happened'"), and
    # nothing in `src/` ever read either one -- `advance` wrote them, `walk` cleared
    # them, and that was all. Three paragraphs of docstring for a mechanism that
    # lived somewhere else entirely: `cycles.Cycle.occurred` and
    # `evidence_within` do that job, per cycle, where the cycle is.

    @property
    def looking_for(self) -> str:
        """The one transition that can legally come next."""
        return PHASES[(PHASES.index(self.curr_stage) + 1) % len(PHASES)]

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
    interrupts=None,
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
        interrupts: the same protocol as ``fires``, used only for the
            out-of-sequence digging alarm. Defaults to the digging STATE (see
            `default_interrupts`) -- or, when ``fires`` is injected, to ``fires``
            itself, so a scripted test drives both with one schedule.
        config: supplies all three as DURATIONS, which is how callers should
            pass them -- the sample counts above exist for tests that want to
            pin an exact number of samples, and override the config when given.

    Returns the detections in the order they were found. Multi-cycle is not
    special-cased: the loop simply keeps going.
    """
    if interrupts is None:
        interrupts = default_interrupts if fires is None else fires
    if fires is None:
        fires = default_fires

    times = np.asarray(table.time_seconds, dtype=float)

    # There is no sample count in this function's defaults, on purpose. A bare `3`
    # would mean 0.30 s on this clip and 0.12 s on a 25 fps clip sampled every
    # frame -- exactly the seconds-versus-indices mixing `Window`'s docstring
    # forbids, tuned to one video's sample spacing. So the durations come from
    # `FSMConfig`, and `samples_for` does the conversion once, here at the boundary.
    #
    # The explicit sample counts remain for tests that need to pin an exact number,
    # and they are read before config so they win. `strict` keeps its documented
    # RATIO to `hold` rather than a duration of its own: "unexpected evidence costs
    # twice as much" is dimensionless and travels between videos.
    # Resolved lazily so a caller that pins every count need not supply a config
    # at all -- and so a test injecting its own trigger is not forced to build one.
    def duration(name: str) -> float:
        source = config.fsm if config is not None else FSMConfig()
        return float(getattr(source, name))

    if hold_samples is None:
        hold_samples = samples_for(duration("hold_seconds"), times)
    if lookback_samples is None:
        lookback_samples = samples_for(duration("lookback_seconds"), times)
    if strict_hold_samples is None:
        # Read from config, not derived as `hold_samples * 2`. Deriving it left
        # `strict_hold_seconds` in `config.py` and `default.yaml` with no reader in
        # `src/` while both files documented it as live -- dead config, the exact
        # defect class this branch has been removing. `FSMConfig`'s own docstring
        # settles which model is right: "Every one is a DURATION, never a count."
        strict_hold_samples = samples_for(duration("strict_hold_seconds"), times)
    strict = strict_hold_samples

    count = len(times)
    # Which samples perception actually saw. A table without `found` -- the
    # scripted tables the sequencing tests inject -- has no gaps to speak of.
    measured = (
        np.asarray(table.found, dtype=bool)
        if hasattr(table, "found")
        else np.ones(count, dtype=bool)
    )
    state = MachineState()
    found: list[Detection] = []
    index = 0
    # Where the previous transition was detected. Every window is clipped to it,
    # so consecutive windows cannot overlap.
    previous = None

    def sustained(
        phase: str,
        start: int,
        needed: int,
        *,
        require_edge: bool = True,
        allow_truncation: bool = True,
        check=None,
    ) -> int:
        """How many samples the trigger held for here, or 0 if this is no transition.

        A rising edge, not a level. The triggers are written as conditions --
        "the bucket is down and still" -- and such a condition is true for the
        whole of the phase it describes, not only at its start. Accepting a
        level would make the always-on digging check fire on every sample of a
        dig it has already recorded: on the development video that produced
        sixteen digging detections inside one digging phase.

        The edge is the whole difference between "digging is happening" and
        "digging just started", and only the second is a transition.

        ``require_edge=False`` is the one exemption, and it exists because phases
        are CONTIGUOUS. Hauling's condition becomes true at or immediately after
        digging's onset is confirmed -- so at the very first sample the walk looks
        for hauling, "already true" is the expected state of affairs and not
        evidence against a transition. Demanding an edge there lost the transition
        permanently, because the condition never goes false again. The exemption
        applies only at that first sample, and only to the phase the walk is
        EXPECTING: an out-of-sequence dig always needs a genuine edge, since
        unexpected evidence should stay expensive.

        The exemption is itself guarded, because unguarded it let already-true
        conditions MARCH the state machine forward on no new evidence: with all four
        triggers true everywhere, the walk produced 20 detections and four
        "complete" cycles out of a signal that never changed. So the exempted phase
        must still have been false at SOME earlier sample -- it must genuinely have
        risen during the clip rather than having been true from the first frame.
        That readmits the contiguous-phase case, where hauling was false right up to
        digging's onset, and refuses the degenerate one, where it was never false at
        all. The rise is not required to be recent: insisting on that is what lost
        the transition in the first place.

        Returns a COUNT rather than a bool so the caller can size the window to
        the evidence that actually exists. At the end of the clip fewer than
        ``needed`` samples remain, and refusing on that ground alone silently
        discarded any transition rising in the last ``needed - 1`` samples -- the
        labelled cycle-closing dig sits at sample 293 of 296 and cleared the old bar
        by exactly zero margin.

        But "some of the evidence" is not "the evidence", so a truncated hold has a
        FLOOR: at least half of what the hold asks for. Without one, a single sample
        at the final index became a detection, and because digging bounds cycles that
        changed `cycle_count` -- the field graded exactly. Half is dimensionless and
        travels; it says a transition must be supported by most of the evidence
        requested, while still surviving a clip that ends one or two frames early.

        ``allow_truncation=False`` refuses any shortfall. The out-of-sequence digging
        check passes it, because that path ABANDONS the cycle in progress and adds a
        boundary: it was accepting a dig on 1 of 6 strict samples, which is the
        opposite of the "unexpected evidence should be expensive" rule it exists to
        enforce.

        ``check`` replaces ``fires`` for this call; the out-of-sequence alarm passes
        ``interrupts``.
        """
        test = fires if check is None else check
        available = count - start
        if available <= 0:
            return 0
        held = min(needed, available)
        if held < needed:
            if not allow_truncation:
                return 0
            if 2 * held < needed:
                return 0
        # The edge is judged against the last sample perception actually SAW, not
        # blindly against `start - 1`. A missing sample makes every trigger return
        # False, so comparing against it turned a one-frame dropout into a rising
        # edge: on the dev clip, one gap mid-dig produced an out-of-sequence dig
        # there and moved the counted cycle's opening onset by seconds. A gap is
        # unknown, not negative -- `_usable` says so -- and that has to hold for
        # the edge as well as for the trigger.
        before = last_measured_before(start)
        if require_edge and before >= 0 and test(phase, table, before, levels, config):
            return 0  # already true before this sample: not an edge
        if not require_edge and before >= 0 and not ever_false_before(phase, start):
            # The exemption is only for a condition that ROSE. One true from the
            # first frame is not evidence of a transition into anything.
            return 0
        if not all(test(phase, table, i, levels, config) for i in range(start, start + held)):
            return 0
        if held < needed:
            log.warning(
                "%s at sample %d held for only %d of %d samples -- the clip ends there. "
                "Accepted on a truncated hold, which is a weaker bar than usual.",
                phase,
                start,
                held,
                needed,
            )
        return held

    # The sample from which the walk started looking for the CURRENT target. The
    # edge rule is relaxed at exactly this sample; see `sustained`.
    looking_since = 0

    def last_measured_before(start: int) -> int:
        """The latest sample before ``start`` that perception saw, or -1 if none."""
        index = start - 1
        while index >= 0 and not measured[index]:
            index -= 1
        return index

    def ever_false_before(phase: str, start: int) -> bool:
        """Was this phase's trigger false at any MEASURED sample before ``start``?

        Scans backward and stops at the first False, so it is cheap in the case that
        matters -- a condition that rose recently answers in one or two calls. Only
        consulted when the edge rule is being waived, which happens at most once per
        detection. Unmeasured samples are skipped for the same reason as in
        `sustained`: a dropout is not the condition going false.
        """
        return any(
            measured[i] and not fires(phase, table, i, levels, config)
            for i in range(start - 1, -1, -1)
        )

    while index < count:
        target = state.looking_for

        # The transition we are expecting.
        held = sustained(target, index, hold_samples, require_edge=index > looking_since)
        if held:
            detected = _detect(state, target, index, held, lookback_samples, times, previous)
            found.append(detected)
            previous = detected.window.hi
            index += held
            looking_since = index
            continue

        # Digging, always, at a higher bar -- it is the only transition that
        # re-establishes where we are.
        #
        # No guard against digging interrupting itself is needed: `sustained`
        # requires a rising edge, and the condition that started a dig stays true
        # for as long as that dig lasts, so it cannot rise again until the bucket
        # has actually come back up. A `curr_stage != "digging"` guard here would
        # be redundant AND wrong -- it would also block a genuine second dig.
        strict_held = sustained(
            "digging", index, strict, allow_truncation=False, check=interrupts
        )
        if target != "digging" and strict_held:
            log.info(
                "digging at sample %d interrupted %s; the cycle being built is abandoned",
                index,
                state.curr_stage,
            )
            state.curr_stage = "swinging"  # so digging is legal again
            state.since = None
            detection = _detect(
                state, "digging", index, strict_held, lookback_samples, times, previous
            )
            found.append(
                Detection(detection.phase, detection.window, detection.fired_at, True)
            )
            previous = detection.window.hi
            index += strict_held
            looking_since = index
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
# Two kinds of question, kept apart
#
# STATE conditions (`in_digging` ...) answer "does this sample look like phase X
# is going on?" They are levels -- "the bucket is down and still" -- so they are
# true for the whole of a phase. That makes them right for the two jobs that ask
# WHETHER a phase happened: the out-of-sequence dig alarm in `walk`, and the
# evidence `cycles.py` counts cycles on (`evidence_within`).
#
# ONSET cues (`trigger_digging` ...) answer "did phase X just start here?" A level
# is poor at that: it was these same conditions, used as onset cues, that missed
# all 13 labelled onsets on the 83 s clip and all 5 on the dev clip. The onset cues
# read a SHAPE instead -- a trend that changes at the moment the phase starts
# (`shapes.py`). The walk only asks for a phase once the previous one has
# started, so a shape that also occurs earlier in the cycle does no harm; what
# matters is that it is the FIRST such shape after the previous onset.
# The evaluation tooling measures exactly that, per cue.
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


def in_digging(table, index: int, levels: Levels) -> bool:
    """STATE: is the bucket digging?  (Formerly the swinging -> digging onset cue.)

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


def in_hauling(table, index: int, levels: Levels) -> bool:
    """STATE: is the bucket hauling?  (Formerly the digging -> hauling onset cue.)

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


def in_dumping(table, index: int, levels: Levels) -> bool:
    """STATE: is the bucket dumping?  (Formerly the hauling -> dumping onset cue.)

    The spec: *dumping begins when the bucket reaches the dumping location and
    starts tipping or uncurling to release its load.* And, crucially: *material
    that spills while the loaded bucket is still being lifted or transported
    remains part of hauling.*

      3.1  the bucket is OVER THE BED -- `truck_overlap` above the level that
           separates "near the truck" from "not". The diagram's "closest to the
           truck".

      3.2  the bucket is OUT PAST THE CABIN, on the side the truck is actually on
           -- `rel_cabin_x * dump_side > 0`. The diagram's "cabin is to the left of
           the bucket" and "higher x value than cabin".

           `dump_side` comes from `calibrate`, which reads it off the footage. The
           earlier version tested `rel_cabin_x > 0`, which is not orientation
           independent -- it IS the orientation, hardcoded. It held on the dev clip
           (86 of 86 over-the-bed samples positive) and would have silently made
           dumping undetectable on any mirrored clip.

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
        and table.rel_cabin_x[index] * levels.dump_side > 0
    )


def in_swinging(table, index: int, levels: Levels) -> bool:
    """STATE: is the machine swinging back?  (Formerly the dumping -> swinging onset cue.)

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


STATES = {
    "digging": in_digging,
    "hauling": in_hauling,
    "dumping": in_dumping,
    "swinging": in_swinging,
}


def _shape(table, index: int, levels: Levels, signal: str, shape: str, *columns: str) -> bool:
    """Does ``signal`` have ``shape`` at this sample? False when unmeasured or unread."""
    if levels.readings is None or not _usable(table, index, *columns):
        return False
    return levels.readings[signal].is_(index, shape)


# Each onset cue below was chosen by a search in the evaluation tooling: of every
# feature and shape, the one whose FIRST occurrence after the previous phase's
# labelled onset was the true onset in the most cycles. Scores are on the 83 s
# clip (three cycles), then the 29 s dev clip (one), as "first match contains the
# true onset within 0.6 s". They were chosen by looking at the 83 s clip, so they
# will score worse on footage they have not seen.


def trigger_digging(table, index: int, levels: Levels) -> bool:
    """ONSET 1  swinging -> digging: the bucket's 2-D speed stops falling and goes flat.

    The return swing decelerates onto the pile; the dig starts where that braking
    ends. Scored 3/3 on the 83 s clip, onset errors -0.3 / -0.6 / -0.0 s.

    WHERE THE SHAPE CANNOT BE READ -- the last seconds of a clip, where there is no
    "after" to fit -- this falls back to the digging STATE (bucket down and still).
    Digging is the cycle boundary, so a dig the cue cannot see deletes a whole cycle
    from `cycle_count`, the one field graded exactly. The dev clip ends that way:
    its bucket is still braking when the video stops, so "drop ends -> flat" never
    happens on camera, and without the fallback its one complete cycle was lost.
    The fallback's timing is the old cue's (-1.23 s there); the count is kept.
    """
    if levels.readings is not None and levels.readings["speed_2d"].shape[index] is None:
        return in_digging(table, index, levels)
    return _shape(table, index, levels, "speed_2d", "drop ends -> flat", "dx_dt", "bucket_y")


def trigger_hauling(table, index: int, levels: Levels) -> bool:
    """ONSET 2  digging -> hauling: height keeps rising.

    The best single shape found, and not yet good enough: 2/3 on the 83 s clip and
    1/1 on the dev clip. Where it misses it fires ~1.5 s EARLY, because the bucket
    starts to climb while still scooping. Gating it on "above the dig level" made
    it LATE in every cycle instead: at the labelled haul onset the bucket is still
    below the level `calibrate` calls "in the material".
    """
    return _shape(table, index, levels, "height", "keeps rising", "height")


def trigger_dumping(table, index: int, levels: Levels) -> bool:
    """ONSET 3  hauling -> dumping: height has been flat and starts rising, on the
    truck's side of the cabin.

    The bucket holds over the bed, then lifts as it uncurls. The best single shape
    found, and not yet good enough: 2/3 on the 83 s clip, and 1.8 s late on the dev
    clip.

    The side gate is what keeps an early haul from cascading. The haul's own lift
    off the pile is ALSO "flat, then rising", so when the haul cue fired early --
    mid-scoop -- the real lift that followed was taken for a dump, and every onset
    after it in that cycle was wrong. A dump can only happen over the truck, and
    `dump_side` says which side of the cabin that is, read from this video.

    Unlike the old overlap gate, this does not need a trustworthy truck level:
    `dump_side` is read from the level as measured. The overlap level was too weak
    to use on the 83 s clip (separability 0.78) and had switched dumping detection
    off there entirely.
    """
    if not _usable(table, index, "rel_cabin_x"):
        return False
    past_cabin = table.rel_cabin_x[index] * levels.dump_side > 0
    return past_cabin and _shape(
        table, index, levels, "height", "flat -> starts rising", "height"
    )


def trigger_swinging(table, index: int, levels: Levels) -> bool:
    """ONSET 4  dumping -> swinging: a peak in horizontal acceleration toward the truck.

    The emptied bucket is first pushed up and out, clear of the bed, before it
    swings back -- that push is the peak. 3/3 on the 83 s clip (errors -0.2 / -0.2
    / -0.3 s) and 1/1 on the dev clip (-0.2 s). It needs the dump onset: scanned
    from the haul onset instead, it fires 10-12 s early, on the haul.

    Signed toward the truck (see `read_signals`), so a mirrored clip reads the same.
    """
    return _shape(table, index, levels, "d2x_toward_truck", "peak", "dx_dt")


TRIGGERS = {
    "digging": trigger_digging,
    "hauling": trigger_hauling,
    "dumping": trigger_dumping,
    "swinging": trigger_swinging,
}


def default_fires(phase: str, table, index: int, levels: Levels, _config=None) -> bool:
    """The onset cue the walk uses when none is injected.

    ``_config`` is part of the injection protocol -- a caller supplying its own
    trigger may want it -- and is unused here, hence the underscore.
    """
    return TRIGGERS[phase](table, index, levels)


def default_interrupts(phase: str, table, index: int, levels: Levels, _config=None) -> bool:
    """The out-of-sequence alarm: the digging STATE, not the digging onset cue.

    The alarm asks "is the machine digging when it should be doing something else?"
    -- a state question. The onset cue's shape (speed levelling off) happens several
    times a cycle, and each one would abandon the cycle in progress.
    """
    return STATES[phase](table, index, levels)


# ---------------------------------------------------------------------------
# Pass 2 -- exactly when, inside a window pass 1 has already chosen
#
# Pass 1 is allowed to be blunt because pass 2 is precise. The division only
# works one way round, though: refine() searches INSIDE the window it is given
# and cannot reach outside it, so a window that misses the transition cannot be
# rescued here. A refined answer from a wrong window is worse than none, because
# it looks precise.
# ---------------------------------------------------------------------------

Mode = Literal["departs", "arrives", "peak", "trigger"]


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
    # THREE states, not two: loud, quiet, and unmeasured. `NaN <= band` is False,
    # so `~quiet` used to call every unmeasured sample MOVING. `departs` then walked
    # backward straight through a gap, and `arrives` could return the time of a
    # sample nobody measured. The same rule as the walk's edge check: a gap is
    # unknown, not evidence -- so it is never crossed and never returned. Only
    # `loud` is needed below; quiet is `finite & ~loud`, and a gap is `~finite`.
    magnitude = np.abs(np.where(finite, inside, 0.0))
    loud = finite & (magnitude > band)

    if mode == "departs":
        # Find the excursion, then walk BACK to where the quiet ended -- stopping at
        # a gap too, and crediting the onset to the first sample that saw motion.
        moving = np.flatnonzero(loud)
        if moving.size == 0:
            return None
        cursor = int(moving[0])
        while cursor > 0 and loud[cursor - 1]:
            cursor -= 1
        return float(times[lo + cursor])

    if mode == "arrives":
        # The true mirror of `departs`: find the excursion, then walk FORWARD to
        # where it ends. Taking the first quiet sample instead returned the window's
        # LEFT EDGE whenever that sample happened to be quiet -- a number produced by
        # the window-clipping rule rather than measured from the signal, and one that
        # looks every bit as precise as a real one.
        moving = np.flatnonzero(loud)
        if moving.size == 0:
            return None  # never moved, so nothing arrived
        # The first MEASURED sample after the last loud one. Every finite sample
        # after `moving[-1]` is quiet by construction, so this is where the signal
        # was first seen at rest; a gap in between is skipped rather than returned.
        settled = np.flatnonzero(finite[int(moving[-1]) + 1 :])
        if settled.size == 0:
            return None  # still moving, or unmeasured, when the window ended
        return float(times[lo + int(moving[-1]) + 1 + int(settled[0])])

    raise ValueError(f"unknown mode {mode!r}")


# Recovering a noise sigma from the low quantile of |successive differences|.
#
# Two corrections, both derived rather than looked up:
#
#   1. |X| for X ~ N(0, s) has its q-th quantile at s * Phi^-1((1+q)/2). So
#      dividing the quantile by that factor inverts it. `inv_cdf` computes it, so
#      changing _QUIET_QUANTILE automatically changes the scale with it.
#   2. Differencing two independent samples of sd s gives sd s*sqrt(2), so the
#      result is divided by sqrt(2) to get back to the per-sample noise.
#
# A LOW quantile rather than the median because a refinement window is mostly
# moving by construction -- see the docstring below.
_QUIET_QUANTILE = 0.25
_QUANTILE_TO_SIGMA = 1.0 / (
    math.sqrt(2.0) * NormalDist().inv_cdf((1.0 + _QUIET_QUANTILE) / 2.0)
)


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

    **A third, found later: the scale factor was for a different statistic.** The
    code multiplied the quantile by ``1.4826 / sqrt(2)``. 1.4826 is the famous
    MAD-to-sigma constant and it is only valid for the MEDIAN of absolute
    deviations -- reaching for it and applying it to the 0.25 quantile recovered
    0.47 sigma where it claimed 1.00, so every band was 2.1x too tight. A band
    that is too tight is the worse direction of error: a settled signal still
    looks like it is moving, so ``arrives`` finds nothing and the onset is LOST
    rather than merely misplaced. The factor below is derived from the quantile
    instead of remembered, so the two cannot drift apart again.
    """
    if values.size < 3:
        return 0.0
    steps = np.abs(np.diff(values))
    quiet = float(np.quantile(steps, _QUIET_QUANTILE)) * _QUANTILE_TO_SIGMA
    return max(sigma * quiet, floor_fraction * float(np.ptp(values)))


# Which signal each transition's onset lives in, and what shape it has.
#
# These are the cues the recovered detector used, and the reason they belong in
# pass 2 as well as pass 1 is that they describe the EVENT rather than the state
# around it. `digging` is the bucket stopping its descent, which is contact;
# `hauling` is the kink where scooping becomes lifting; `swinging` is the machine
# leaving rest; `dumping` is the bucket's silhouette at its most stretched, which
# is the closest thing available to "it has tipped".
@dataclass(frozen=True)
class Onset:
    """One transition, with what each pass was able to say about it.

    Both times are kept on purpose. `refined` is pass 2's answer and is the only
    one fit to appear in a duration; `coarse` is pass 1's trigger time and is
    always available. Keeping only `refined` is what made a failed cue delete a
    cycle BOUNDARY, and with it the cycle, from a video where the cycle plainly
    happened -- `cycle_count` went to 0 on a clip holding one complete cycle.

    So the rule is: `coarse` bounds the span a cycle is asked about, `refined`
    measures it. A phase can be counted on the strength of the first while being
    excluded from the averages for want of the second, which is exactly the
    counting-versus-measuring split `cycles.py` is built around.
    """

    phase: str
    refined: float | None  # pass 2's instant; None when its cue found nothing
    coarse: float  # pass 1's trigger time; always present
    out_of_sequence: bool = False


def evidence_within(table, levels, start: float, end: float) -> set[str]:
    """Which phases left ANY trace between two times. The weak second check.

    Weaker than an onset by design, and that is the whole point: a phase that left
    a trace but whose onset could not be pinned still happened.

    Be precise about HOW MUCH weaker, because it is less than it sounds. This runs
    the SAME four triggers, with both of their conditions still ANDed -- dumping
    still requires the bucket to be on the truck's side of the cabin, and still
    returns False without a usable truck level. What is dropped is only the HOLD and
    the RISING EDGE: one satisfying sample anywhere in the span counts, where an
    onset needs a rise followed by evidence that persists.

    So it does not implement "the bucket was over the bed at some point" -- an
    earlier version of this docstring claimed that, and it is a stronger promise than
    the code keeps.

    This is the mechanism `cycles.py` documented from the beginning and did not
    have. Without it the caller has to guess, and the CLI guessed badly: it built
    one set from the whole video and copied it into every cycle, so one dumping
    detection anywhere marked every cycle as having dumped.
    """
    times = np.asarray(table.time_seconds, dtype=float)
    # HALF-OPEN, matching `Window`. With both ends inclusive the closing digging
    # onset's sample belonged to this cycle AND was the next cycle's opening sample,
    # so a single dumping sample sitting exactly on a boundary counted as evidence in
    # both -- enough to mark the wrong cycle complete and inflate `cycle_count`.
    inside = np.flatnonzero((times >= start) & (times < end))
    # `for_evidence` restores a truck level that was too weak to pin an onset. A
    # gate disabled for measurement must not also erase the record that the phase
    # happened -- see `Levels.for_evidence`.
    permissive = levels.for_evidence
    return {
        phase
        for phase in PHASES
        if any(STATES[phase](table, int(i), permissive) for i in inside)
    }


# How pass 2 turns each detection into an instant. ``"trigger"`` means pass 1
# already measured it: a shape cue fires where its shape STARTS, which is the onset
# itself, so there is nothing left to refine. It is not the coarse guess the old
# level cues produced -- the evaluation tooling scores these times directly.
# Dumping keeps a real refinement: the bucket box at its most stretched, as it
# tips, moved the first dump on the 83 s clip from +0.61 s to -0.19 s.
REFINEMENTS: dict[str, tuple[str, Mode]] = {
    "digging": ("speed_x", "trigger"),
    "hauling": ("height", "trigger"),
    "dumping": ("aspect_ratio", "peak"),
    "swinging": ("dx_dt", "trigger"),
}


def locate(detections: list[Detection], table, config) -> list[Onset]:
    """Turn pass 1's windows into instants: the join between the two passes.

    A detection whose cue finds nothing keeps its place in the sequence, with
    ``refined=None``. It is emphatically NOT given the trigger time as its onset:
    a made-up onset would flow into a duration and be indistinguishable from a
    measured one. But dropping the detection outright was worse, because
    `assemble` splits cycles on digging, so a dropped digging refinement deleted a
    cycle BOUNDARY -- and the cycle with it. On the dev clip that turned one
    complete, fully-detected cycle into ``cycle_count: 0``.

    `Onset` keeps both answers so the caller can count on the coarse time and
    measure only on the refined one. A failure is now visible in the sequence
    rather than absent from it, which is what `Cycle.reason` needs to explain it.

    Ordering needs no enforcement here -- pass 1's windows cannot overlap, so a
    refined time cannot cross its neighbour. That is checked by a test rather
    than asserted at runtime, because if it ever breaks the symptom is a
    negative phase duration rather than an exception.
    """
    times = np.asarray(table.time_seconds, dtype=float)
    out: list[Onset] = []
    for detection in detections:
        column, mode = REFINEMENTS[detection.phase]
        if mode == "trigger":
            out.append(
                Onset(
                    phase=detection.phase,
                    refined=float(times[detection.fired_at]),
                    coarse=float(times[detection.fired_at]),
                    out_of_sequence=detection.out_of_sequence,
                )
            )
            continue
        when = refine(
            getattr(table, column),
            times,
            detection.window,
            mode,
            sigma=config.fsm.rest_sigma,
            floor_fraction=config.fsm.rest_floor_fraction,
        )
        if when is None:
            log.warning(
                "%s at sample %d: the %s cue found no %s in [%d, %d). The cycle is "
                "still counted; this phase cannot be measured.",
                detection.phase,
                detection.fired_at,
                column,
                mode,
                detection.window.lo,
                detection.window.hi,
            )
        out.append(
            Onset(
                phase=detection.phase,
                refined=when,
                coarse=float(times[detection.fired_at]),
                out_of_sequence=detection.out_of_sequence,
            )
        )
    return out
