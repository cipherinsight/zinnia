"""The old entry point of the phase 1 sweep, kept so that a command written
against step 1 or step 2 still runs.

The sweep is :mod:`testing.phase1.sweep`; this module only forwards to it.
"""

from __future__ import annotations

from testing.phase1.sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
