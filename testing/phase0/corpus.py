"""The phase 0 parity set: every benchmark circuit the baseline compiles.

`revision/phase0/coverage.md` §1 defines the set as the circuits recorded
`compile_ok` in two sweep records under `benchmarking/`. This module reads
those records, loads the kernels the way the benchmark harness does
(`benchmarking/harness/kernels.py`, which replaced
`benchmarking/validate_contracts.py`), and hands back `ZKCircuit` objects.

It also holds the child-process isolation every phase 0 harness uses
(:func:`run_in_child`) and the coverage harness's own worker
(:func:`analyze_isolated`), which is how a harness stays alive when a program
unrolls further than the machine allows.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from testing.phase0 import REPO_ROOT, record

BENCHMARKING = REPO_ROOT / "benchmarking"

#: The two sweep records of `coverage.md` §1, family name -> record path.
RECORDS: dict[str, Path] = {
    "ds1000": BENCHMARKING
    / "archive"
    / "2026-06-agent-sweeps"
    / "ds1000_numpy_compile_validation_2026-05-22_after_op_contract.json",
    "npbench": BENCHMARKING / "archive" / "2026-06-agent-sweeps" / "compile_validation.json",
}

#: Where each program family's folders live since the datasets moved under
#: `benchmarking/datasets/`; the records keep their original relative paths.
DATASET_DIRS: dict[str, Path] = {
    "ds1000": BENCHMARKING / "datasets" / "ds1000",
    "npbench": BENCHMARKING / "datasets" / "npbench",
    "pythran": BENCHMARKING / "datasets" / "pythran",
}


def dataset_path(record_path: str) -> Path:
    """Resolve a record's `kernel_path` (e.g. `npbench_x/kernel.py`) to its folder."""
    family = record_path.split("_", 1)[0]
    return DATASET_DIRS[family] / record_path

#: The status a record row must carry to be in the parity set.
ACCEPTED_STATUS = "compile_ok"


@dataclass(frozen=True)
class Circuit:
    """One `@zk_circuit` function of the parity set."""

    family: str
    program: str
    function: str
    kernel_path: Path

    @property
    def label(self) -> str:
        return f"{self.program}::{self.function}"


def load_module(path: Path) -> ModuleType:
    """One kernel file, imported the way the benchmark harness imports it."""
    from benchmarking.harness.kernels import load_kernel_module

    return load_kernel_module(path)


def function_roles(path: Path) -> dict[str, str]:
    """Which functions of a kernel file are circuits and which are chips."""
    from benchmarking.harness.kernels import function_roles as harness_function_roles

    return harness_function_roles(path)


def _family_of(program: str, record: str) -> str:
    """The reporting family: `coverage.md` §1 splits NPBench from Pythran."""
    if record == "ds1000":
        return "ds1000"
    return program.split("_", 1)[0]


def _rows(record: str) -> list[dict[str, Any]]:
    payload = json.loads(RECORDS[record].read_text(encoding="utf-8"))
    return payload["results"]


def parity_set() -> list[Circuit]:
    """Every circuit of `coverage.md` §1's parity set, in record order."""
    circuits: list[Circuit] = []
    for record in RECORDS:
        for row in _rows(record):
            if row.get("status") != ACCEPTED_STATUS:
                continue
            circuits.append(
                Circuit(
                    family=_family_of(row["program"], record),
                    program=row["program"],
                    function=row["function"],
                    kernel_path=dataset_path(row["kernel_path"]),
                )
            )
    return circuits


def outside_parity_set() -> Counter:
    """How many circuits of each record the baseline does not compile.

    `acceptance.md` §1: programs outside the parity set are reported, not
    counted. The counter is keyed by the record's status string.
    """
    counts: Counter = Counter()
    for record in RECORDS:
        for row in _rows(record):
            if row.get("role") != "circuit" or row.get("status") == ACCEPTED_STATUS:
                continue
            counts[row.get("status", "unknown")] += 1
    return counts


def load_function(circuit: Circuit) -> Any:
    """The decorated circuit function of one parity entry."""
    module = load_module(circuit.kernel_path)
    function = getattr(module, circuit.function, None)
    if function is None:
        raise AttributeError(
            f"{circuit.function} not found in {circuit.kernel_path}"
        )
    return function


def load_zk_circuit(circuit: Circuit) -> Any:
    """The `ZKCircuit` for one parity entry, chips and externals included."""
    from zinnia import ZKCircuit

    return ZKCircuit.from_method(load_function(circuit))

# ---------------------------------------------------------------------------
# Running one job in its own process
# ---------------------------------------------------------------------------
#
# Phase 0's work budget (`implementation.md` §8) is the first line of defence
# against a program whose unrolled form does not fit in memory, but it counts
# work, not bytes: a single unit can allocate a descriptor, a term and a
# variable. So the coverage, adequacy and consistency harnesses all run their
# per-circuit job in a child process with a hard address-space limit and a
# timeout, the way `validate_contracts.compile_one` isolates a compile. The
# parent never does the work itself, so a child that is killed costs one row
# of the report instead of the run.
#
# The isolation is here, once; each harness supplies its own
# :class:`ChildLimits` and its own worker.


@dataclass(frozen=True)
class ChildLimits:
    """One harness's isolation budget."""

    #: Address space the child may use (`resource.RLIMIT_AS`).
    address_space_bytes: int
    #: Wall-clock seconds the child may take.
    timeout_seconds: float

    @property
    def address_space_gib(self) -> int:
        return self.address_space_bytes >> 30


@dataclass(frozen=True)
class ChildResult:
    """What a child answered, or why it did not.

    `status` is `ok` (the worker put `answer` on the queue), `timeout` (it
    outlived its budget), `memory_exceeded` (it died of a signal the kernel
    sends when memory runs out) or `crashed` (it exited with no answer for
    another reason). `detail` says which, in words a report can print.
    """

    status: str
    answer: dict[str, Any] | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


#: Address space one phase 0 analysis may use in the coverage harness.
ADDRESS_SPACE_LIMIT = 6 * 1024**3

#: Wall-clock seconds one analysis may take.
ANALYZE_TIMEOUT_SECONDS = 120.0

#: The coverage harness's budget.
ANALYZE_LIMITS = ChildLimits(ADDRESS_SPACE_LIMIT, ANALYZE_TIMEOUT_SECONDS)


def limit_address_space(limit_bytes: int) -> None:
    """Cap this process's address space. Called first in every worker.

    A hard limit lower than `limit_bytes` is already what the caller wanted,
    so failing to raise it is not an error.
    """
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        pass


def run_in_child(
    target: Callable[..., None], args: tuple, limits: ChildLimits
) -> ChildResult:
    """Run `target(*args, queue)` in a spawned child and collect its answer.

    The worker is expected to put exactly one dict on `queue`, whatever
    happened, and to cap its own address space with
    :func:`limit_address_space` before it allocates anything.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=target, args=(*args, queue))
    child.start()
    child.join(limits.timeout_seconds)
    if child.is_alive():
        _kill(child)
        return ChildResult(
            "timeout", detail=f"exceeded {limits.timeout_seconds:g}s"
        )
    if not queue.empty():
        return ChildResult("ok", answer=queue.get())
    # SIGKILL is the cgroup's OOM killer; SIGABRT is a Rust allocation
    # failure under `RLIMIT_AS`, which prints "memory allocation of N bytes
    # failed" and aborts.
    if child.exitcode in (-9, -6, 137):
        return ChildResult(
            "memory_exceeded",
            detail=(
                f"the child died with signal {-(child.exitcode or 0)} under an "
                f"address-space limit of {limits.address_space_gib} GiB"
            ),
        )
    return ChildResult(
        "crashed",
        detail=f"the child exited with code {child.exitcode} and no answer",
    )


def _kill(child: Any) -> None:
    """Stop a child that outlived its budget, escalating to `SIGKILL`."""
    child.terminate()
    child.join(2)
    if child.is_alive():
        child.kill()
        child.join()


def _analyze_worker(kernel_path: str, function: str, queue) -> None:
    """Analyse one circuit and put the summary of its record on `queue`."""
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from zinnia import ZKCircuit

        module = load_module(Path(kernel_path))
        result = ZKCircuit.from_method(getattr(module, function)).analyze_shapes()
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        queue.put(
            {
                "status": "rejected",
                "reason": "harness_" + type(exc).__name__,
                "detail": first_line(exc, 300),
            }
        )
        return
    rejection = result.get("rejected")
    if rejection is not None:
        queue.put(
            {
                "status": "rejected",
                "reason": rejection.get("reason"),
                "detail": rejection.get("detail"),
            }
        )
        return
    # The record invariants of `implementation.md` §5a, checked here because
    # this is the one place every accepted circuit passes through.
    broken = record.violations(result)
    if broken:
        queue.put(
            {
                "status": "invariant_violation",
                "reason": "record_invariant",
                "detail": "; ".join(broken)[:300],
            }
        )
        return
    # Only the sizes cross the process boundary: the record itself is large
    # and nothing in the coverage report reads it.
    queue.put(
        {
            "status": "ok",
            "variables": len(result["variables"]),
            "constraints": len(result["constraints"]),
            "terms": len(result["terms"]["nodes"]),
            "descriptors": len(result["descriptors"]),
            "record": record.summary(result),
        }
    )


def first_line(exc: BaseException, limit: int) -> str:
    """An exception reduced to one printable line, for a report row."""
    return (str(exc).splitlines() or [repr(exc)])[0][:limit]


def analyze_isolated(
    circuit: Circuit,
    timeout_seconds: float = ANALYZE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Phase 0's answer for one circuit, as a summary dict.

    The `status` is `ok` (phase 0 returned a record), `rejected` (it returned
    a rejection, with `reason` and `detail`), or one of
    :class:`ChildResult`'s failure statuses.
    """
    limits = ChildLimits(ADDRESS_SPACE_LIMIT, timeout_seconds)
    outcome = run_in_child(
        _analyze_worker, (str(circuit.kernel_path), circuit.function), limits
    )
    if outcome.answer is not None:
        return outcome.answer
    return {
        "status": outcome.status,
        "reason": outcome.status,
        "detail": outcome.detail,
    }
