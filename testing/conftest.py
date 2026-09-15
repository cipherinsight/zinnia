"""Opt-in phase 0 adequacy checking over the whole test suite.

`revision/phase0/acceptance.md` §2.6 asks for the adequacy check to run on the
programs the suites under `testing/` execute. It is off unless
`ZINNIA_PHASE0_ADEQUACY=1` is set, and when it is off this file does nothing at
all: no import of the harness, no change to any test.

```sh
ZINNIA_PHASE0_ADEQUACY=1 .venv/bin/python -m pytest testing/ -q
```

This conftest sits at `testing/` rather than at `testing/phase0/` because the
corpus it serves is the rest of the suite; pytest applies a conftest only to
the directory it lives in and below.
"""

from __future__ import annotations

import os
from typing import Any

#: Set by `pytest_configure` when the hook is enabled.
_HOOK: Any = None


def pytest_configure(config: Any) -> None:
    del config
    if os.environ.get("ZINNIA_PHASE0_ADEQUACY", "") in ("", "0", "false", "no"):
        return
    from testing.phase0.adequacy_hook import install_if_enabled

    global _HOOK
    _HOOK = install_if_enabled()


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    del session, exitstatus
    if _HOOK is None:
        return
    _HOOK.uninstall()
    path = _HOOK.write()
    print(f"\n{_HOOK.report.summary()}\nwrote {path}")
