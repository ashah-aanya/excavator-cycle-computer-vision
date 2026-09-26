"""Cycles, and the answer. Design diagram stage 4.

Counting and measuring are different questions
----------------------------------------------
A cycle the machine *performed* is complete whether or not the pipeline managed
to pin all four of its onsets. So this module keeps two populations:

* ``cycle_count`` counts what **happened**, and
* the averages come from what could be **measured**.

The spec supports the split: it asks to *identify every complete work cycle* and,
separately, for *the average duration of each phase across all complete cycles*.
A cue failing is a fact about the pipeline; it is not a fact about the excavator.

One consequence is worth stating loudly, because it looks like a bug:
``cycle_count * average_cycle_duration_seconds`` will **not** equal the elapsed
time whenever a cycle occurred but could not be measured. That is correct. The
two numbers answer different questions and come from different populations.

Telling "we missed a cue" from "no cycle happened" needs a weaker second check
than an onset: did the phase leave *any* evidence at all? If the bucket was over
the bed at some point in the span, dumping happened and the cue failed. If it
never went near the truck, the machine re-dug and no cycle occurred. That check
is ``fsm.evidence_within``, and it is asked about **each cycle's own span** --
passing one set built from the whole video, as an earlier version did, marks every
cycle complete as soon as any cycle anywhere was.

The split only survives if it is respected upstream too. ``locate`` keeps a
detection whose pass-2 cue failed, with ``refined=None``, because ``assemble``
splits on digging: dropping such a detection deletes a cycle BOUNDARY and takes
the cycle out of *both* populations. A span is therefore bounded by pass 1's
coarse times, which always exist, and measured from pass 2's refined ones, which
may not.

Head and tail partials need no rule
-----------------------------------
A cycle runs from one digging onset to the next, so footage before the first and
after the last falls outside every cycle and is ignored -- which is the spec's
*"ignore any incomplete cycle at the beginning or end"* without a special case.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .fsm import Onset
from .logging_setup import get_logger

log = get_logger(__name__)

PHASES = ("digging", "hauling", "dumping", "swinging")

# Which phases left any trace between two times. See `fsm.evidence_within`.
Evidence = Callable[[float, float], set[str]]


@dataclass(frozen=True)
class Cycle:
    """One digging onset to the next.

    ``onsets`` holds whichever of the four were REFINED; ``occurred`` holds
    whichever left evidence, refined or not. The gap between those two sets is
    what separates a failed cue from an absent phase.

    ``span`` is the coarse pair of trigger times that bound this cycle. It exists
    separately from ``onsets`` because it must be available even when nothing
    refined -- it is what the evidence check is asked about, and what makes a
    cycle countable when it is not measurable.
    """

    onsets: dict[str, float]
    ends: float | None  # the next REFINED digging onset; None if it was not found
    span: tuple[float, float]
    occurred: set[str] = field(default_factory=set)

    @property
    def complete(self) -> bool:
        """Did the machine perform all four phases, however well we timed them?"""
        return all(phase in self.occurred for phase in PHASES)

    @property
    def measurable(self) -> bool:
        """Can this cycle contribute a duration?

        Stricter than ``complete``: every onset located, in order, and a closing
        onset to measure the last phase against.
        """
        return self.reason is None

    @property
    def reason(self) -> str | None:
        """Why this cycle cannot be measured, or ``None`` when it can.

        A sentence rather than a flag: when a hidden video yields two cycles
        instead of four, the useful question is which two were rejected and
        what for.
        """
        absent = [p for p in PHASES if p not in self.occurred]
        if absent:
            return f"no {', '.join(absent)} in this span -- no cycle occurred"
        missing = [p for p in PHASES if p not in self.onsets]
        if missing:
            return f"{', '.join(missing)} occurred but its onset was not located"
        if self.ends is None:
            return "the closing digging onset occurred but was not located"
        ordered = [self.onsets[p] for p in PHASES] + [self.ends]
        if any(b < a for a, b in pairwise(ordered)):
            return f"onsets are out of order: {[round(v, 2) for v in ordered]}"
        return None

    @property
    def duration(self) -> float | None:
        """Digging onset to the next digging onset."""
        if self.ends is None or "digging" not in self.onsets:
            return None
        return self.ends - self.onsets["digging"]

    def durations(self) -> dict[str, float]:
        """Each phase's length. Empty when the cycle is not measurable.

        Swinging runs to the NEXT digging onset, per the spec: it *ends
        immediately before the next digging phase begins*.
        """
        if not self.measurable:
            return {}
        edges = [self.onsets[p] for p in PHASES] + [self.ends]
        return {p: edges[i + 1] - edges[i] for i, p in enumerate(PHASES)}


@dataclass(frozen=True)
class Answer:
    """Exactly the schema the task asks for, and nothing else."""

    cycle_count: int
    average_cycle_duration_seconds: float
    average_phase_duration_seconds: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "cycle_count": int(self.cycle_count),
            "average_cycle_duration_seconds": round(
                float(self.average_cycle_duration_seconds), 3
            ),
            "average_phase_duration_seconds": {
                p: round(float(v), 3) for p, v in self.average_phase_duration_seconds.items()
            },
        }


def assemble(onsets: list[Onset], evidence: Evidence | None = None) -> list[Cycle]:
    """Turn a run of onsets into cycles.

    Splits on digging, because that is the cycle boundary. N digging onsets
    bound N-1 cycles, and anything outside the first and last is in no cycle --
    the spec's head-and-tail rule, for free.

    Args:
        evidence: ``(start_seconds, end_seconds) -> set[str]`` -- which phases
            left any trace in this cycle's span. Asked per cycle, never once for
            the video. ``fsm.evidence_within`` is the real one; omitting it falls
            back to "the phases that were detected here", which is weaker but
            still per-cycle.
    """
    digs = [i for i, onset in enumerate(onsets) if onset.phase == "digging"]
    cycles: list[Cycle] = []
    for start, stop in pairwise(digs):
        inside = onsets[start:stop]
        # Coarse times always exist, so the span always exists.
        bounds = (inside[0].coarse, onsets[stop].coarse)
        located: dict[str, float] = {}
        for onset in inside:
            if onset.refined is not None:
                located.setdefault(onset.phase, onset.refined)  # the FIRST per span
        occurred = evidence(*bounds) if evidence is not None else {o.phase for o in inside}
        cycles.append(
            Cycle(
                onsets=located,
                ends=onsets[stop].refined,
                span=bounds,
                occurred=set(occurred),
            )
        )
    return cycles


def summarise(cycles: list[Cycle]) -> Answer:
    """Count what occurred; average what could be measured.

    Zero measurable cycles is a legitimate outcome -- a clip shorter than one
    cycle has nothing to average -- and is reported as zeros rather than raised,
    so the pipeline still produces a file a reviewer can read.
    """
    counted = [c for c in cycles if c.complete]
    measured = [c for c in cycles if c.measurable]

    if not measured:
        log.warning(
            "%d cycle(s) occurred but none could be measured; the averages are zero",
            len(counted),
        )
        return Answer(len(counted), 0.0, dict.fromkeys(PHASES, 0.0))

    if len(measured) < len(counted):
        log.info(
            "%d of %d cycles are measurable; the rest are counted but excluded "
            "from the averages",
            len(measured),
            len(counted),
        )

    per_phase = {
        phase: sum(c.durations()[phase] for c in measured) / len(measured) for phase in PHASES
    }
    mean_cycle = sum(c.duration for c in measured) / len(measured)
    return Answer(len(counted), mean_cycle, per_phase)


def write_answer(answer: Answer, path: str | Path) -> Path:
    """Write ``answer.json``."""
    path = Path(path)
    path.write_text(json.dumps(answer.to_dict(), indent=2) + "\n")
    log.info("wrote %s", path)
    return path
