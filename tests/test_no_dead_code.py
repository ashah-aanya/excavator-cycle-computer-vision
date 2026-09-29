"""Every definition in the package must have a real caller.

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

Why it is stricter than it was
------------------------------
The first version counted SUBSTRING occurrences, and that was too crude to work.
A review found `MachineState.elapsed` dead while this test passed, because the
word "elapsed" appears in `cycles.py`'s module docstring -- changing that one word
of prose made the gate fail. The same collision hid `onsets.motion_boundary` (a
docstring mentioning its own name) and `onsets.smooth` (the word "smooth" in a
comment in `boxes.py`). A gate that prose can switch off is not a gate.

It also walked only `FunctionDef`/`ClassDef`, so it could not see dataclass
FIELDS -- and `MachineState.pending`/`.occurred` were exactly the original defect
in its second form: three paragraphs about "the weak second check", written by
`advance`, cleared by `walk`, and read by nothing in `src/`.

So references are now resolved from the AST (`Name`, `Attribute`, decorators,
keyword arguments, import aliases) and definitions include annotated class-level
assignments. Comments and docstrings count for nothing, which is the point.

Adding to ALLOWED is a decision, not a formality. It means "this is reachable in
a way the AST scan cannot see", and the reason goes next to it.
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "excavator_cycles"

# Reachable in ways an AST scan over `src/` cannot see. Each entry says how.
ALLOWED = {
    # Entry points: the console script and `python -m excavator_cycles`.
    "main",
    # Called by the CLI through `args.func`, never by name.
    "_cmd_probe",
    "_cmd_track",
    "_cmd_render",
    "_cmd_features",
    "_cmd_cycles",
    "_cmd_run",
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


def _definitions() -> dict[str, Path]:
    """Every function, class and dataclass field in the package, except dunders."""
    found: dict[str, Path] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if not node.name.startswith("__"):
                    found.setdefault(node.name, path)
            # An annotated assignment; at class level this is a dataclass field.
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and not node.target.id.startswith("__")
            ):
                found.setdefault(node.target.id, path)
    return found


def _references(path: Path) -> collections.Counter[str]:
    """Every name this file actually REFERS to, from the AST rather than the text.

    Deliberately excludes comments, docstrings and every other string constant,
    because counting those is what let dead code hide behind prose.
    """
    counts: collections.Counter[str] = collections.Counter()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition_sites: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definition_sites.add(id(node))
        if isinstance(node, ast.Name):
            counts[node.id] += 1
        elif isinstance(node, ast.Attribute):
            counts[node.attr] += 1
        elif isinstance(node, ast.keyword) and node.arg:
            counts[node.arg] += 1
        elif isinstance(node, ast.ImportFrom | ast.Import):
            for alias in node.names:
                counts[alias.asname or alias.name.split(".")[-1]] += 1
    return counts


def test_every_definition_in_the_package_is_referenced_somewhere_in_it():
    """A definition nothing references is either dead or a docstring's promise.

    Both are worth failing a build over: the first is weight, and the second is
    a lie that a reader will believe.
    """
    references: collections.Counter[str] = collections.Counter()
    for path in PACKAGE.rglob("*.py"):
        references.update(_references(path))

    orphans = [
        f"{defined_in.relative_to(PACKAGE.parent.parent)}::{name}"
        for name, defined_in in _definitions().items()
        if name not in ALLOWED and references[name] == 0
    ]

    assert not orphans, (
        "nothing in src/ references these:\n  "
        + "\n  ".join(sorted(orphans))
        + "\n\nEither wire it up, delete it, or add it to ALLOWED with a reason. "
        "A definition with no caller and a docstring describing what it does is "
        "how `note_occurred` came to document a mechanism that did not exist."
    )


def test_the_gate_ignores_prose():
    """The regression that made this file stricter.

    A name mentioned only in a comment or docstring must not count as a reference.
    Without this, `elapsed` stayed hidden for a whole branch behind one word in an
    unrelated module's docstring.
    """
    import tempfile

    source = '''
"""A docstring mentioning nonexistent_helper and also elapsed."""
# A comment mentioning nonexistent_helper too.
MESSAGE = "nonexistent_helper appears in this string as well"
'''
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        path = Path(handle.name)
    try:
        assert _references(path)["nonexistent_helper"] == 0, (
            "prose and string constants must not count as references"
        )
        assert _references(path)["MESSAGE"] == 1, "a real assignment does count"
    finally:
        path.unlink()


def test_the_gate_sees_dataclass_fields():
    """`pending` and `occurred` were fields, and the first version of this gate
    walked only functions and classes -- so it could not see the defect it exists
    to prevent, in the form it actually took the second time."""
    import tempfile

    source = (
        "import dataclasses\n\n\n@dataclasses.dataclass\nclass Thing:\n    a_field: int = 0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=PACKAGE) as handle:
        handle.write(source)
        path = Path(handle.name)
    try:
        assert "a_field" in _definitions(), (
            "an annotated class-level assignment is a definition"
        )
    finally:
        path.unlink()
