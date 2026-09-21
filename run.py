#!/usr/bin/env python3
"""Entry point for the pipeline.

Thin shim so the project can be run as `uv run run.py ...` without installing
anything, which is how a reviewer will most likely try it first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from excavator_cycles.cli import main

if __name__ == "__main__":
    sys.exit(main())
