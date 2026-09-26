"""Every public thing in the package must have a caller.

Why this file exists
--------------------
`MachineState.note_occurred` was written with three paragraphs of docstring
describing it as the mechanism that separates "the cue failed" from "no cycle
happened". It was called from exactly one place in the repository: a test. The
behaviour its docstring described was never implemented, and the module that
depended on it silently used a different, wrong rule instead.

Nothing caught that. The test suite could not: a test cannot notice the absence
of a call it was never asked to make. Review found it, eventually, after later
work had been built on top.

This file is the cheap check that would have caught it the same afternoon. It is
deliberately crude -- a symbol is "used" if its name appears anywhere else in
`src/` -- because the failure it guards against is total absence, not subtle
misuse, and a crude check that runs is worth more than a precise one that does
not.

Adding to ALLOWED is a decision, not a formality. It means "this is reachable in
a way the grep cannot see", and the reason goes next to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "excavator_cycles"

# Reachable in ways a grep over `src/` cannot see. Each entry says how.
ALLOWED = {
    # Entry points: the console script and `python -m excavator_cycles`.
    "main",
    # Called by the CLI through `args.func`, never by name.
    "_cmd_probe",
    "_cmd_track",
    "_cmd_render",
    "_cmd_features",
    "_cmd_cycles",
    # Public API a reviewer or a notebook uses directly, by design.
    "probe",
    "render",
    "load",
    "save",
    "load_result",
    "load_objects",
    "save_objects",
    "encode",
    "decode",
    # Dataclass/protocol surface: read by callers as attributes, not called.
    "to_dict",
    "from_dict",
    "summary",
    "describe",
    "report",
    "columns",
    "resolution",
}


def _public_definitions() -> dict[str, Path]:
    """Every top-level and method name defined in the package, except dunders."""
    found: dict[str, Path] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name.startswith("__"):
                    continue
                found.setdefault(node.name, path)
    return found


def test_every_definition_in_the_package_is_referenced_somewhere_in_it():
    """A definition nothing references is either dead or a docstring's promise.

    Both are worth failing a build over: the first is weight, and the second is
    a lie that a reader will believe.
    """
    sources = {path: path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")}
    orphans = []
    for name, defined_in in _public_definitions().items():
        if name in ALLOWED:
            continue
        # "Used" means the name appears in some OTHER file, or more than once in
        # its own (the definition itself being the first occurrence).
        uses = sum(
            text.count(name) - (1 if path == defined_in else 0)
            for path, text in sources.items()
        )
        if uses == 0:
            orphans.append(f"{defined_in.relative_to(PACKAGE.parent.parent)}::{name}")

    assert not orphans, (
        "nothing in src/ references these:\n  "
        + "\n  ".join(sorted(orphans))
        + "\n\nEither wire it up, delete it, or add it to ALLOWED with a reason. "
        "A definition with no caller and a docstring describing what it does is "
        "how `note_occurred` came to document a mechanism that did not exist."
    )
