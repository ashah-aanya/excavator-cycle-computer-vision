"""Cycles, and the answer.

A cycle runs from one digging start to the next: dig, haul, dump, swing, and the next
dig closes it. Footage before the first digging start and after the last falls outside
every cycle and is ignored, which is the spec's *"ignore any incomplete cycle at the
beginning or end"* without a special case.

A cycle counts if the dig that opens it, the dig that closes it and at least
`MIN_PHASES_FOUND` phases in between (the opening dig included) were found, in order.
Missing one phase does not throw the cycle away: an excavator that dug, and then showed
two more of haul, dump and swing before digging again, went through a cycle. A span with
fewer than that is dropped. The count and the averages come from the same set of cycles.

A phase lasts from its start to the next phase's start, and swinging runs to the next
digging start, per the spec: it *ends immediately before the next digging phase begins*.
A phase is only measured where both its start and the start of the phase after it were
found. If the dump is missing, the haul has no end and the dump has no start, so neither
is measured for that cycle; the cycle's own duration (dig to dig) always is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from .logging_setup import get_logger
from .starts import PhaseSearch, PhaseStart

log = get_logger(__name__)

PHASES = ("digging", "hauling", "dumping", "swinging")

# The opening dig plus two of the other three. Two is the least that says "a cycle
# happened" rather than "the search glimpsed one phase": a dig and a lone haul could be
# any partial motion.
MIN_PHASES_FOUND = 3


@dataclass(frozen=True)
class Cycle:
    """One counted cycle. Times are seconds from the start of the video."""

    starts: dict[str, float]  # each phase's start; a phase the search missed is absent
    end: float  # the next digging start: where this cycle's swinging ends

    @property
    def duration(self) -> float:
        return self.end - self.starts["digging"]

    def durations(self) -> dict[str, float]:
        """The length of each phase that can be measured: its own start and the start of
        the next phase (or the cycle's end, for swinging) must both have been found."""
        out = {}
        for i, phase in enumerate(PHASES):
            if phase not in self.starts:
                continue
            if i + 1 == len(PHASES):
                out[phase] = self.end - self.starts[phase]
            elif PHASES[i + 1] in self.starts:
                out[phase] = self.starts[PHASES[i + 1]] - self.starts[phase]
        return out


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


def assemble(starts: list[PhaseStart]) -> list[Cycle]:
    """Turn the phase starts, in the order they were found, into complete cycles.

    Splits on digging, because that is the cycle boundary: N digging starts bound at
    most N-1 cycles. A span counts when its phases are in cycle order and at least
    `MIN_PHASES_FOUND` were found; any other span is dropped and logged.
    """
    digs = [i for i, start in enumerate(starts) if start.phase == "digging"]
    cycles = []
    for first, following in pairwise(digs):
        inside = starts[first:following]
        found = [start.phase for start in inside]
        in_order = [PHASES.index(p) for p in found] == sorted({PHASES.index(p) for p in found})
        if len(found) < MIN_PHASES_FOUND or not in_order:
            log.info(
                "the span from %.1f s has %s, not at least %d phases in order; not counted",
                inside[0].time,
                found,
                MIN_PHASES_FOUND,
            )
            continue
        if len(found) < len(PHASES):
            log.info(
                "the span from %.1f s is missing %s; counted, and those phases are not "
                "measured for it",
                inside[0].time,
                [p for p in PHASES if p not in found],
            )
        cycles.append(Cycle({s.phase: s.time for s in inside}, starts[following].time))
    return cycles


def summarise(cycles: list[Cycle]) -> Answer:
    """Count the cycles and average their durations.

    The cycle duration averages every counted cycle. A phase's average uses only the
    cycles where that phase could be measured, so a cycle with a missing phase does not
    pull that phase's average toward zero.

    No complete cycle is a legitimate outcome -- a clip shorter than one cycle has
    nothing to average -- and is reported as zeros rather than raised, so the pipeline
    still produces a file a reviewer can read.
    """
    if not cycles:
        log.warning("no complete cycle was found; the averages are zero")
        return Answer(0, 0.0, dict.fromkeys(PHASES, 0.0))
    per_phase = {}
    for phase in PHASES:
        measured = [c.durations()[phase] for c in cycles if phase in c.durations()]
        if not measured:
            log.warning("%s was measured in no cycle; its average is zero", phase)
        per_phase[phase] = sum(measured) / len(measured) if measured else 0.0
    return Answer(len(cycles), sum(c.duration for c in cycles) / len(cycles), per_phase)


def write_answer(answer: Answer, path: str | Path) -> Path:
    """Write ``answer.json``."""
    path = Path(path)
    path.write_text(json.dumps(answer.to_dict(), indent=2) + "\n")
    log.info("wrote %s", path)
    return path


def write_phases(search: PhaseSearch, cycles: list[Cycle], path: str | Path) -> Path:
    """Write what the phase search found, so the annotated video draws exactly that."""
    record = {
        "starts": [asdict(start) for start in search.starts],
        "cycles": [{"starts": c.starts, "end": c.end} for c in cycles],
        "gaps": search.gaps,
        "stop": search.stop,
    }
    path = Path(path)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def read_phases(path: str | Path) -> tuple[list[PhaseStart], list[Cycle]]:
    """Read back what :func:`write_phases` wrote."""
    record = json.loads(Path(path).read_text())
    starts = [PhaseStart(**{**s, "window": tuple(s["window"])}) for s in record["starts"]]
    return starts, [Cycle(c["starts"], c["end"]) for c in record["cycles"]]
