"""Allow `python -m excavator_cycles ...`."""

import sys

from .cli import main

sys.exit(main())
