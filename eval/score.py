"""Score an ``answer.json`` against the hand-made ground truth in ``eval/labels.json``.

**This is evaluation scaffolding, not pipeline code.** It lives outside ``src/`` on
purpose: the task spec forbids the pipeline from reading the answer, so the labels
must be somewhere the pipeline cannot reach them even by accident. Nothing under
``src/excavator_cycles/`` may import this module, and this module imports nothing
from the pipeline -- standard library only, so it stays runnable even in an
environment where the pipeline's dependencies are not installed.

Usage::

    uv run python eval/score.py outputs/answer.json
    python eval/score.py outputs/answer.json --labels eval/labels.json

Exit code is 0 when every field passes and 1 otherwise -- including when the
answer file is missing or malformed, because "we could not read your answer" is a
failure, not a crash. No traceback ever reaches the user.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LABELS = Path(__file__).resolve().parent / "labels.json"

PHASES = ("digging", "hauling", "dumping", "swinging")


class ScoringError(Exception):
    """Something about the input files made scoring impossible.

    Raised rather than printed so that the one place that knows how to talk to the
    user -- ``main`` -- stays the one place that does.
    """


# --------------------------------------------------------------------------- read


def _load_json(path: Path, what: str) -> dict:
    """Read a JSON object, turning every plausible failure into a clear message."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ScoringError(f"{what} not found: {path}") from None
    except IsADirectoryError:
        raise ScoringError(f"{what} is a directory, not a file: {path}") from None
    except OSError as exc:
        raise ScoringError(f"could not read {what} at {path}: {exc}") from None
    except UnicodeDecodeError:
        raise ScoringError(f"{what} at {path} is not UTF-8 text") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScoringError(
            f"{what} at {path} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from None

    if not isinstance(data, dict):
        raise ScoringError(
            f"{what} at {path} must be a JSON object, got {type(data).__name__}"
        )
    return data


def _number(value: object, field: str, source: str) -> float:
    """Coerce a JSON value to a float, or explain why it is not one.

    ``bool`` is rejected explicitly: in Python it is a subclass of ``int``, so
    ``True`` would otherwise silently score as 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(
            f"{source}: field {field!r} must be a number, got {value!r} "
            f"({type(value).__name__})"
        )
    return float(value)


def _integer(value: object, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoringError(
            f"{source}: field {field!r} must be an integer, got {value!r} "
            f"({type(value).__name__})"
        )
    return value


def extract_fields(data: dict, source: str) -> dict[str, float]:
    """Pull the five graded values out of an answer-shaped object.

    The same extraction runs over the answer and over the labels' own
    ``expected_answer`` block, so the two can never drift apart in how they are
    read. Phase values are flattened to dotted keys for the table.
    """
    if "cycle_count" not in data:
        raise ScoringError(f"{source}: missing required field 'cycle_count'")
    fields: dict[str, float] = {
        "cycle_count": float(_integer(data["cycle_count"], "cycle_count", source))
    }

    if "average_cycle_duration_seconds" not in data:
        raise ScoringError(
            f"{source}: missing required field 'average_cycle_duration_seconds'"
        )
    fields["average_cycle_duration_seconds"] = _number(
        data["average_cycle_duration_seconds"], "average_cycle_duration_seconds", source
    )

    phases = data.get("average_phase_duration_seconds")
    if phases is None:
        raise ScoringError(
            f"{source}: missing required field 'average_phase_duration_seconds'"
        )
    if not isinstance(phases, dict):
        raise ScoringError(
            f"{source}: 'average_phase_duration_seconds' must be an object mapping "
            f"phase name to seconds, got {type(phases).__name__}"
        )
    for phase in PHASES:
        key = f"average_phase_duration_seconds.{phase}"
        if phase not in phases:
            raise ScoringError(f"{source}: missing required field {key!r}")
        fields[key] = _number(phases[phase], key, source)
    return fields


def load_expected(labels_path: Path = DEFAULT_LABELS) -> dict[str, float]:
    """The ground-truth values, read from the labels file's ``expected_answer`` block."""
    labels = _load_json(labels_path, "labels file")
    expected = labels.get("expected_answer")
    if not isinstance(expected, dict):
        raise ScoringError(
            f"labels file at {labels_path} has no 'expected_answer' object; "
            "it is not a labels file this scorer understands"
        )
    return extract_fields(expected, f"labels file {labels_path.name}")


def load_tolerance(labels_path: Path = DEFAULT_LABELS) -> float:
    """Tolerance in seconds, taken from the labels rather than hardcoded here."""
    labels = _load_json(labels_path, "labels file")
    scoring = labels.get("scoring")
    if not isinstance(scoring, dict) or "tolerance_seconds" not in scoring:
        raise ScoringError(f"labels file at {labels_path} has no 'scoring.tolerance_seconds'")
    return _number(scoring["tolerance_seconds"], "scoring.tolerance_seconds", "labels file")


# -------------------------------------------------------------------------- score


class Result:
    """One field's verdict. A tiny class so the table and the exit code agree."""

    def __init__(
        self, field: str, expected: float, actual: float, passed: bool, *, exact: bool
    ):
        self.field = field
        self.expected = expected
        self.actual = actual
        self.passed = passed
        self.exact = exact  # counted, not timed: no tolerance applies

    @property
    def error(self) -> float:
        return self.actual - self.expected


def score(
    expected: dict[str, float], actual: dict[str, float], tolerance: float
) -> list[Result]:
    """Compare field by field.

    ``cycle_count`` is compared exactly: it is a count of cycles, and being "0.6
    cycles off" is not a thing. Everything else is a duration in seconds and
    passes when the absolute error is within tolerance.
    """
    results = []
    for field, want in expected.items():
        got = actual[field]
        exact = field == "cycle_count"
        passed = (got == want) if exact else (abs(got - want) <= tolerance)
        results.append(Result(field, want, got, passed, exact=exact))
    return results


# -------------------------------------------------------------------------- report


def format_table(results: list[Result], tolerance: float) -> str:
    """A fixed-width table. Readability beats machine-parseability here: a human
    reads this to find out *which* field drifted and in *which direction*, so the
    error column is signed."""
    width = max(len(r.field) for r in results)
    lines = [
        f"{'field':<{width}}  {'expected':>9}  {'actual':>9}  {'error':>8}  result",
        "-" * (width + 42),
    ]
    for r in results:
        if r.exact:
            expected_s = f"{r.expected:.0f}"
            actual_s = f"{r.actual:.0f}"
            error_s = "exact" if r.passed else "differs"
        else:
            expected_s = f"{r.expected:.3f}"
            actual_s = f"{r.actual:.3f}"
            error_s = f"{r.error:+.3f}"
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.field:<{width}}  {expected_s:>9}  {actual_s:>9}  {error_s:>8}  {verdict}"
        )
    lines.append("-" * (width + 42))
    overall = "PASS" if all(r.passed for r in results) else "FAIL"
    lines.append(
        f"OVERALL: {overall}   (tolerance +/-{tolerance:.3f} s per field; "
        "cycle_count must match exactly)"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="score.py",
        description=(
            "Score an answer.json against the hand-made ground truth in eval/labels.json. "
            "Evaluation only -- the pipeline must never read these labels."
        ),
    )
    parser.add_argument("answer", type=Path, help="path to the answer.json to score")
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="path to the ground-truth labels (default: eval/labels.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="override the per-field tolerance in seconds (default: read from the labels)",
    )
    args = parser.parse_args(argv)

    try:
        expected = load_expected(args.labels)
        tolerance = (
            args.tolerance if args.tolerance is not None else load_tolerance(args.labels)
        )
        answer = _load_json(args.answer, "answer file")
        actual = extract_fields(answer, f"answer file {args.answer}")
    except ScoringError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("OVERALL: FAIL   (could not score this answer)", file=sys.stderr)
        return 1

    results = score(expected, actual, tolerance)
    print(f"answer: {args.answer}")
    print(f"labels: {args.labels}")
    print()
    print(format_table(results, tolerance))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
