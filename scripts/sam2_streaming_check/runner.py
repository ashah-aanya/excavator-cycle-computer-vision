"""TEMPORARY (fix/sam2-streaming) -- delete before merging.

Run the pipeline once, in this process, and record what the streaming PR needs to
know about the run: exit status, wall time and peak GPU memory.

    runner.py <label> <mode> <out_dir> -- <run.py arguments...>

mode:
  stream   the new tracker (the default config)
  offline  the previous tracker, via `track.streaming: false`
  keepall  the new tracker with output pruning disabled (window effectively
           infinite) -- the "pruned == kept" control for stage 3

Peak memory is `torch.cuda.max_memory_allocated`: what this process's tensors
actually held at their highest, on the GPU the run used. It is the number the
old tracker blew through (a 9.75 GiB request) and the one streaming must keep
flat.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from excavator_cycles import track
from excavator_cycles.cli import main

HERE = Path(__file__).resolve().parent


def run(label: str, mode: str, out_dir: Path, argv: list[str]) -> int:
    if mode == "keepall":
        track.output_window = lambda _model: 10**9
    elif mode == "offline":
        # A one-line YAML layered over the defaults; `run` passes it to every stage.
        argv = [*argv, "--config", str(HERE / "offline.yaml")]
    elif mode != "stream":
        raise SystemExit(f"unknown mode {mode!r}")

    started = time.perf_counter()
    try:
        status = main(argv)
        error = None
    except Exception as exc:  # recorded, not swallowed: the summary reports it
        status, error = 99, f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started

    cuda = torch.cuda.is_available()
    record = {
        "label": label,
        "mode": mode,
        "exit": status,
        "error": error,
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0) if cuda else "no cuda",
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if cuda
        else None,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)
        if cuda
        else None,
    }
    (out_dir / f"{label}.peak.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    if error:
        raise SystemExit(error)
    return status


if __name__ == "__main__":
    label, mode, out = sys.argv[1:4]
    if sys.argv[4:5] != ["--"]:
        raise SystemExit("usage: runner.py <label> <mode> <out_dir> -- <run.py args>")
    sys.exit(run(label, mode, Path(out), sys.argv[5:]))
