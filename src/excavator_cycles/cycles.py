"""Cycles, and the answer.

A cycle runs from one digging start to the next: dig, haul, dump, swing, and the next
dig closes it. Footage before the first digging start and after the last falls outside
every cycle and is ignored, which is the spec's *"ignore any incomplete cycle at the
beginning or end"* without a special case.

A cycle counts only if all four phases were found inside it, in order. The phase
search records a cycle it had to abandon (a phase it could not find) by restarting at
the next dig, so such a span has fewer than four phases here and is dropped. The
count and the averages therefore come from the same set of cycles.

A phase lasts from its start to the next phase's start, and swinging runs to the next
digging start, per the spec: it *ends immediately before the next digging phase begins*.
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


@dataclass(frozen=True)
class Cycle:
    """One complete cycle. Times are seconds from the start of the video."""

    starts: dict[str, float]  # each phase's start
    end: float  # the next digging start: where this cycle's swinging ends

    @property
    def duration(self) -> float:
        return self.end - self.starts["digging"]

    def durations(self) -> dict[str, float]:
        """Each phase's length."""
        edges = [self.starts[p] for p in PHASES] + [self.end]
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


def assemble(starts: list[PhaseStart]) -> list[Cycle]:
    """Turn the phase starts, in the order they were found, into complete cycles.

    Splits on digging, because that is the cycle boundary: N digging starts bound at
    most N-1 cycles. A span that does not hold exactly dig, haul, dump, swing is
    dropped and logged.
    """
    digs = [i for i, start in enumerate(starts) if start.phase == "digging"]
    cycles = []
    for first, following in pairwise(digs):
        inside = starts[first:following]
        if [start.phase for start in inside] != list(PHASES):
            log.info(
                "the span from %.1f s has %s, not all four phases; not counted",
                inside[0].time,
                [start.phase for start in inside],
            )
            continue
        cycles.append(Cycle({s.phase: s.time for s in inside}, starts[following].time))
    return cycles


def summarise(cycles: list[Cycle]) -> Answer:
    """Count the complete cycles and average their durations.

    No complete cycle is a legitimate outcome -- a clip shorter than one cycle has
    nothing to average -- and is reported as zeros rather than raised, so the pipeline
    still produces a file a reviewer can read.
    """
    if not cycles:
        log.warning("no complete cycle was found; the averages are zero")
        return Answer(0, 0.0, dict.fromkeys(PHASES, 0.0))
    per_phase = {p: sum(c.durations()[p] for c in cycles) / len(cycles) for p in PHASES}
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
