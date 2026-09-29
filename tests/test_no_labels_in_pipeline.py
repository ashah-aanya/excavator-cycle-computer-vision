"""The task's two bans on reading answers, enforced instead of promised.

The pipeline may not read `answer.json`, and it may not hardcode or read a video's
answers. Hand labels exist only to SCORE a result, from outside the pipeline, so nothing
under `src/` may name them, and nothing under `src/` may read an answer file back in.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "excavator_cycles"

LABEL_NAMES = ("eval/labels", "labels.json", "labels_long_clip", "eval.score", "check_cues")


def test_pipeline_never_references_the_labels():
    """No module under src/ may name the label files, the scorer or the cue checker."""
    offenders = [
        str(path.relative_to(PACKAGE.parent.parent))
        for path in PACKAGE.rglob("*.py")
        if any(name in path.read_text(encoding="utf-8") for name in LABEL_NAMES)
    ]
    assert not offenders, f"pipeline modules reference the eval labels: {offenders}"


def test_pipeline_never_reads_an_answer_file():
    """`answer.json` is written by the pipeline and never read back by it. Any call that
    reads a file, with an argument that mentions an answer, is a way to break the rule."""
    reading = {"read_text", "read_bytes", "open", "load", "loads", "loadtxt"}
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in reading and "answer" in ast.unparse(node).lower():
                offenders.append(f"{path.name}:{node.lineno}  {ast.unparse(node)[:70]}")
    assert not offenders, f"the pipeline reads an answer: {offenders}"
