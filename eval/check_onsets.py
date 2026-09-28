"""Run the pipeline's state machine on a clip and compare every onset to the labels.

**Evaluation scaffolding, not pipeline code.** ``check_cues.py`` scores one
candidate cue on paper; this runs the real thing -- ``calibrate``, ``walk`` and
``locate`` from ``excavator_cycles.fsm``, exactly as the CLI does -- and says, for
every labelled onset, what the pipeline found and how far off it was.

The dependency points one way only. This file imports the pipeline; nothing in the
pipeline may import or name anything in ``eval/``
(``tests/test_eval_labels.py::test_pipeline_never_references_the_labels``).

Usage::

    uv run python eval/check_onsets.py                 # the 83 s clip
    uv run python eval/check_onsets.py --clip dev      # the 29 s dev clip

For each labelled onset it prints the pipeline's detection of that phase nearest
in time: pass 1's trigger time (``coarse``) and pass 2's refined time, each with
its error. Exit code 0 when every refined onset is within the tolerance, 1 when
one is not, 2 when the inputs cannot be read.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent


def _check_cues():
    """Reuse check_cues' clip table and label reader rather than a second copy."""
    spec = importlib.util.spec_from_file_location("check_cues", EVAL / "check_cues.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def table_from(path: Path):
    """A FeatureTable straight from features.npz -- no scene.json needed."""
    from excavator_cycles.features import FeatureTable

    z = np.load(path)
    return FeatureTable(**{f.name: z[f.name] for f in dataclasses.fields(FeatureTable)})


def run_pipeline(table, config):
    from excavator_cycles.fsm import calibrate, locate, walk

    levels = calibrate(table, config)
    detections = walk(table, levels, config=config)
    return locate(detections, table, config)


def compare(onsets, labelled, tolerance):
    """For each labelled onset, the nearest detected onset of the same phase."""
    rows = []
    for phase, truth in labelled:
        same = [o for o in onsets if o.phase == phase]
        if not same:
            rows.append((phase, truth, None, None, False))
            continue
        best = min(
            same, key=lambda o: abs((o.refined if o.refined is not None else o.coarse) - truth)
        )
        ok = best.refined is not None and abs(best.refined - truth) <= tolerance
        rows.append((phase, truth, best.coarse, best.refined, ok))
    return rows


def _err(value, truth):
    return "     --" if value is None else f"{value - truth:+6.2f} s"


def main(argv: list[str] | None = None) -> int:
    cc = _check_cues()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", choices=sorted(cc.CLIPS), default="long")
    ap.add_argument("--features", type=Path)
    ap.add_argument("--labels", type=Path)
    args = ap.parse_args(argv)

    from excavator_cycles.config import Config

    try:
        feat_path = args.features or cc.CLIPS[args.clip][0]
        label_path = args.labels or cc.CLIPS[args.clip][1]
        labelled = cc.load_onsets(label_path)
        tolerance = cc.tolerance_of(label_path)
        table = table_from(feat_path)
    except (cc.CheckError, OSError, KeyError, ValueError) as exc:
        print(f"check_onsets: {exc}", file=sys.stderr)
        return 2

    onsets = run_pipeline(table, Config.load(EVAL.parent / "configs" / "default.yaml"))
    rows = compare(onsets, labelled, tolerance)

    print(f"{feat_path.parent.name}: pipeline onsets vs labels (tolerance {tolerance:g} s)\n")
    print("  phase      labelled    coarse err   refined err   within?")
    for phase, truth, coarse, refined, ok in rows:
        print(
            f"  {phase:<9} {truth:7.2f} s  {_err(coarse, truth)}     "
            f"{_err(refined, truth)}      {'yes' if ok else 'NO'}"
        )
    n_ok = sum(r[4] for r in rows)
    print(f"\n  {n_ok}/{len(rows)} labelled onsets matched within {tolerance:g} s")
    found = ", ".join(
        f"{o.phase}@{o.coarse:.1f}{'!' if o.out_of_sequence else ''}" for o in onsets
    )
    print(f"  pipeline found {len(onsets)} onsets: {found or 'none'}")
    print("  (! = an out-of-sequence dig that abandoned the cycle in progress)")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
