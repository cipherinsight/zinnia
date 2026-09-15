"""The second adequacy corpus: the test suites under `testing/`.

`revision/phase0/acceptance.md` §2.6. The suites construct circuits inline and
call them with concrete inputs and expected results, so they are a cheaper
place to diagnose an adequacy failure than a benchmark kernel. This module
wraps `ZKCircuit.__call__` — which is what `ZKCircuit.mock` and every
`circuit(...)` in the suites go through — so that each mock execution also
runs phase 0 and the reference evaluator on the same inputs and applies the
check of §2.5.

The hook is **opt-in**: `testing/conftest.py` installs it only when
`ZINNIA_PHASE0_ADEQUACY=1` is set, so an ordinary `pytest testing/` run is
untouched.

```sh
ZINNIA_PHASE0_ADEQUACY=1 .venv/bin/python -m pytest testing/ -q \\
    --ignore=testing/phase0
```

Nothing the hook observes can fail a test: a rejection, a harness gap or a
mismatch is counted and written to `revision/phase0/results/adequacy_hook.json`
at the end of the session. The suites test the baseline, not phase 0.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from testing.phase0 import RESULTS

#: The environment variable that turns the hook on.
ENV_VAR = "ZINNIA_PHASE0_ADEQUACY"

#: Where the session summary is written.
HOOK_JSON = RESULTS / "adequacy_hook.json"


@dataclass
class HookReport:
    """What the hook saw over a whole pytest session."""

    executions: int = 0
    checked: int = 0
    rejected: Counter = field(default_factory=Counter)
    gaps: Counter = field(default_factory=Counter)
    harness_errors: Counter = field(default_factory=Counter)
    succeeded: int = 0
    failed: int = 0
    passed: int = 0
    classes: Counter = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "executions": self.executions,
            "checked": self.checked,
            "rejected": dict(self.rejected),
            "gaps": dict(self.gaps.most_common(20)),
            "harness_errors": dict(self.harness_errors.most_common(20)),
            "reference_runs": {"succeeded": self.succeeded, "failed": self.failed},
            "passed": self.passed,
            "mismatch_classes": dict(self.classes),
            "examples": self.examples[:60],
        }

    def summary(self) -> str:
        return (
            f"phase 0 adequacy hook: {self.executions} mock executions, "
            f"{self.checked} checked, {self.passed} passed; "
            f"{sum(self.rejected.values())} rejected, "
            f"{sum(self.gaps.values())} harness gaps, "
            f"{sum(self.harness_errors.values())} harness errors; "
            f"mismatches {dict(self.classes) or '{}'}"
        )


class AdequacyHook:
    """Wraps `ZKCircuit.__call__` and checks every execution it sees."""

    def __init__(self) -> None:
        self.report = HookReport()
        self._records: dict[tuple[str, str], Any] = {}
        self._original: Callable[..., Any] | None = None

    # ── installation ──────────────────────────────────────────────────

    def install(self) -> "AdequacyHook":
        from zinnia import ZKCircuit

        if self._original is not None:
            return self
        self._original = ZKCircuit.__call__
        hook = self

        def patched(circuit: Any, *args: Any) -> Any:
            hook.observe(circuit, list(args))
            return hook._original(circuit, *args)  # noqa: SLF001 - saved bound fn

        ZKCircuit.__call__ = patched
        return self

    def uninstall(self) -> None:
        from zinnia import ZKCircuit

        if self._original is not None:
            ZKCircuit.__call__ = self._original
            self._original = None

    # ── the check ─────────────────────────────────────────────────────

    def observe(self, circuit: Any, args: list[Any]) -> None:
        """Run phase 0 and the reference evaluator on one mock execution.

        Every failure mode is counted, never raised: the suites are testing
        the baseline and must behave the same with the hook installed.
        """
        self.report.executions += 1
        try:
            prepared = self._prepare(circuit)
        except BaseException as exc:  # PyO3 panics inherit from BaseException
            self.report.harness_errors[f"{type(exc).__name__}: {exc}"[:120]] += 1
            return
        if prepared is None:
            return
        record, payloads, impls = prepared
        try:
            self._check_once(record, payloads, impls, args, circuit)
        except BaseException as exc:
            self.report.harness_errors[f"{type(exc).__name__}: {exc}"[:120]] += 1

    def _prepare(self, circuit: Any) -> tuple[Any, tuple, dict] | None:
        """Phase 0's record for this circuit, analysed once per source."""
        from testing.phase0.adequacy import Phase0Record
        from testing.phase0.payloads import core_payloads, external_impls

        identity = (circuit.name, circuit.source)
        if identity not in self._records:
            analysis = circuit.analyze_shapes()
            if "rejected" in analysis:
                self._records[identity] = None
                reason = analysis["rejected"].get("reason")
                self.report.rejected[f"{circuit.name}: {reason}"] += 1
            else:
                self._records[identity] = (
                    Phase0Record.of(analysis), core_payloads(circuit),
                    external_impls(circuit),
                )
        return self._records[identity]

    def _check_once(
        self, record: Any, payloads: tuple, impls: dict, args: list[Any], circuit: Any
    ) -> None:
        from testing.phase0.adequacy import check
        from testing.phase0.refeval import evaluate

        name = circuit.name
        trace = evaluate(payloads, args, impls)
        if trace.gap is not None:
            self.report.gaps[f"{name}: {trace.gap}"[:160]] += 1
            return
        verdict = check(record, trace)
        self.report.checked += 1
        self.report.succeeded += int(verdict.succeeded)
        self.report.failed += int(not verdict.succeeded)
        self.report.passed += int(verdict.passed)
        for mismatch in verdict.mismatches:
            self.report.classes[mismatch.classification] += 1
            if len(self.report.examples) < 200:
                self.report.examples.append(dict(
                    mismatch.as_dict(), circuit=name,
                    source=_body(circuit.source),
                ))

    # ── reporting ─────────────────────────────────────────────────────

    def write(self) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        HOOK_JSON.write_text(
            json.dumps(self.report.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return HOOK_JSON


def _body(source: str) -> str:
    """The circuit's body, squeezed onto one line.

    Suite circuits are almost all called `foo`, so the name alone does not say
    which test a mismatch came from; the body does.
    """
    lines = [line.strip() for line in source.splitlines()
             if line.strip() and not line.strip().startswith(("@", "#"))]
    return " ".join(lines)[:400]


def enabled() -> bool:
    """Whether `ZINNIA_PHASE0_ADEQUACY` asks for the hook."""
    return os.environ.get(ENV_VAR, "") not in ("", "0", "false", "no")


def install_if_enabled() -> AdequacyHook | None:
    """The one call `testing/conftest.py` makes."""
    return AdequacyHook().install() if enabled() else None
