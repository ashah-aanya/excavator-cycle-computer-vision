"""Logging configuration.

Why not ``print``: the perception stage takes minutes on a GPU node, and when
something looks wrong afterwards the only evidence is what was written while it
ran. Logging gives timestamps, module names and severity levels, can be turned
up without editing code, and can be sent to a file alongside the run's outputs
-- which is exactly what you want when the run happened on a cluster an hour ago.

Convention used throughout this package:

    DEBUG    per-sample detail (every frame, every detection)
    INFO     stage progress a human would want to watch
    WARNING  something that may distort the answer, but the run continues
    ERROR    the stage cannot produce its output

The WARNING level is load-bearing here. This pipeline must always produce an
``answer.json`` -- failing loudly is not an option on the hidden videos -- so
degraded conditions are warnings plus a flag in the QA report, never exceptions.
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbosity: int = 0, log_file: str | Path | None = None) -> None:
    """Set up logging for a run.

    Args:
        verbosity: 0 = INFO, 1+ = DEBUG. Wire this to repeated ``-v`` flags.
        log_file: if given, also write the full DEBUG trail here, regardless of
            the console level. Console stays readable; the file keeps everything
            for when you need it later.
    """
    level = logging.DEBUG if verbosity > 0 else logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter; the root must let it through
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root.addHandler(file_handler)

    # Third-party libraries are chatty and rarely say anything we need. `httpx`
    # is the loud one: newer huggingface_hub routes downloads through it, and it
    # logs every redirect and cache probe at INFO, burying the pipeline's own
    # output in HTTP traffic.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
        "huggingface_hub",
        "transformers",
        "PIL",
        "matplotlib",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Call as ``get_logger(__name__)``."""
    return logging.getLogger(name)
