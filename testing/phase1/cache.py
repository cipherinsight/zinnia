"""The per-circuit cache every phase 1 measurement reads
(`revision/phase1/acceptance.md` §1).

The parity set is run once: for each circuit, phase 0 and phase 1 in a child
process with a memory cap and a timeout, and one JSON file per circuit
holding the phase 0 record, the phase 1 result and the wall-clock timings.
Every table of `revision/phase1/results/REPORT.md` is computed from these
files, so the corpus is never re-run to answer a new question, and a circuit
is re-run only when the code that produced its file changed (`--force`, or
the schema tag of the file no longer matching).

One file per circuit:

```text
{
  "schema":   "zinnia.phase1.cache/1",
  "label":    "program::function", "family": str, "program": str,
  "function": str, "kernel_path": str,
  "status":   "ok" | "phase0_rejected" | "phase1_rejected" | "timeout"
              | "memory_exceeded" | "crashed" | "harness",
  "reason":   str | null,  "detail": str | null,
  "seconds":  {"analyze": float, "candidates": float},
  "phase0":   the record of `src/phase0/export.rs` (no pretty fields) | null,
  "phase1":   the record of `src/phase1/export.rs` | null
}
```

`seconds.analyze` is `analyze_shapes` alone, which is phase 0;
`seconds.candidates` is `build_candidates`, which runs phase 0 again and then
phase 1, so phase 1's own wall clock is the difference of the two, to within
one phase 0 run's variance. The phase 1 record carries the oracle's own
solver time in `stats.solver.millis`, which is the number the questions
table reports.

Run it from `zinnia-src`:

```text
MEMRUN_MAX=20G <memrun.sh> env PYTHON=.venv/bin/python \\
  .venv/bin/python -m testing.phase1.cache --all
```

`--only TEXT` (repeatable) restricts the run to the circuits whose label
contains `TEXT`; `--force` re-runs a circuit whose file exists. The cache
directory defaults to :data:`CACHE_DIR` and is `--cache` or
`ZINNIA_PHASE1_CACHE`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from testing.phase0 import REPO_ROOT
from testing.phase0.corpus import (
    ChildLimits,
    Circuit,
    first_line,
    limit_address_space,
    load_module,
    parity_set,
    run_in_child,
)
from testing.phase0.terms import TermDag

#: The schema of a cache file; bump it when the layout above changes, so
#: that a stale file is re-run rather than read.
SCHEMA = "zinnia.phase1.cache/1"

#: The phase 1 export schema this harness reads (`src/phase1/export.rs`).
PHASE1_SCHEMA = "zinnia.phase1/5"

#: Where the cache lives unless `--cache` or `ZINNIA_PHASE1_CACHE` says
#: otherwise: under the crate's build directory, which is ignored by git and
#: is where a measurement artefact of this size belongs.
CACHE_DIR = Path(os.environ.get("ZINNIA_PHASE1_CACHE",
                                str(REPO_ROOT / "target" / "phase1-cache")))

#: Address space one circuit's child may use: the sweep's 6 GiB plus room
#: for serialising the two records.
ADDRESS_SPACE_LIMIT = 10 * 1024**3

#: Wall-clock seconds one circuit may take, both phases and the write.
TIMEOUT_SECONDS = 300.0

#: The oracle's limits, as `sweep.py` sets them.
QUERY_TIMEOUT_MS = 3000
PROGRAM_TIMEOUT_MS = 60_000


# ---------------------------------------------------------------------------
# Reading the cache
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """One circuit's cache file, loaded."""

    label: str
    family: str
    program: str
    function: str
    status: str
    reason: str | None
    detail: str | None
    seconds: dict[str, float]
    phase0: dict[str, Any] | None
    phase1: dict[str, Any] | None
    path: Path

    @property
    def accepted(self) -> bool:
        return self.status == "ok"

    @property
    def rejected_by_phase1(self) -> bool:
        return self.status == "phase1_rejected"

    def terms(self) -> TermDag:
        """The one term arena phase 0 and phase 1 share
        (`revision/phase1/README.md` §1.1): phase 0's nodes followed by the
        nodes phase 1 built, so that every term id an answer, a condition
        or a requirement names resolves.
        """
        if self.phase0 is None:
            raise ValueError(f"{self.label}: no phase 0 record")
        nodes = list(self.phase0["terms"]["nodes"])
        if self.phase1 is not None and "terms" in self.phase1:
            built = self.phase1["terms"]
            if int(built["base"]) != len(nodes):
                raise ValueError(
                    f"{self.label}: phase 1's terms start at {built['base']}, "
                    f"phase 0's arena has {len(nodes)} nodes"
                )
            nodes.extend(built["nodes"])
        return TermDag(nodes=nodes, vars=TermDag.from_json(self.phase0["terms"]).vars)

    def checks(self) -> list[tuple[int, dict[str, Any]]]:
        """Φ's checks in check order: (constraint index, constraint)."""
        if self.phase0 is None:
            return []
        return [
            (at, constraint)
            for at, constraint in enumerate(self.phase0["constraints"])
            if constraint["kind"] == "check"
        ]

    @staticmethod
    def load(path: Path) -> "Entry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Entry(
            label=payload["label"],
            family=payload["family"],
            program=payload["program"],
            function=payload["function"],
            status=payload["status"],
            reason=payload.get("reason"),
            detail=payload.get("detail"),
            seconds=payload.get("seconds") or {},
            phase0=payload.get("phase0"),
            phase1=payload.get("phase1"),
            path=path,
        )


def file_of(circuit: Circuit, cache: Path = CACHE_DIR) -> Path:
    """The cache file of one circuit."""
    return cache / f"{circuit.program}__{circuit.function}.json"


def is_current(path: Path) -> bool:
    """Whether a cache file exists and carries the schema this code writes."""
    if not path.exists():
        return False
    with path.open("rb") as handle:
        head = handle.read(200)
    return SCHEMA.encode() in head


def entries(cache: Path = CACHE_DIR,
            circuits: Iterable[Circuit] | None = None) -> Iterator[Entry]:
    """The cache, one entry at a time, in parity-set order.

    A circuit whose file is missing is skipped; :func:`missing` says which.
    The entries are yielded rather than listed because the largest records
    are tens of megabytes each and the measurements read them one by one.
    """
    for circuit in circuits if circuits is not None else parity_set():
        path = file_of(circuit, cache)
        if is_current(path):
            yield Entry.load(path)


def missing(cache: Path = CACHE_DIR) -> list[Circuit]:
    """The parity circuits with no current cache file."""
    return [circuit for circuit in parity_set() if not is_current(file_of(circuit, cache))]


def size_bytes(cache: Path = CACHE_DIR) -> int:
    return sum(path.stat().st_size for path in cache.glob("*.json"))


# ---------------------------------------------------------------------------
# Filling the cache: one circuit in its own process
# ---------------------------------------------------------------------------


def _worker(kernel_path: str, function: str, out_path: str, meta: dict[str, Any],
            queue) -> None:
    """Both phases for one circuit; the file is written here, in the child,
    so that a record of tens of megabytes never crosses the process
    boundary through the queue."""
    limit_address_space(ADDRESS_SPACE_LIMIT)
    sys.path.insert(0, str(REPO_ROOT))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        **meta,
        "status": "harness",
        "reason": None,
        "detail": None,
        "seconds": {},
        "phase0": None,
        "phase1": None,
    }
    try:
        from zinnia import ZKCircuit

        module = load_module(Path(kernel_path))
        circuit = ZKCircuit.from_method(getattr(module, function))
        started = time.perf_counter()
        record = circuit.analyze_shapes(pretty=False)
        payload["seconds"]["analyze"] = time.perf_counter() - started
    except BaseException as exc:  # PyO3 panics inherit from BaseException
        payload.update(reason=type(exc).__name__, detail=first_line(exc, 300))
        _write(out_path, payload, queue)
        return
    rejection = record.get("rejected")
    if rejection is not None:
        payload.update(status="phase0_rejected", reason=rejection["reason"],
                       detail=rejection.get("detail", ""))
        _write(out_path, payload, queue)
        return
    payload["phase0"] = record
    try:
        started = time.perf_counter()
        result = circuit.build_candidates(QUERY_TIMEOUT_MS, PROGRAM_TIMEOUT_MS)
        payload["seconds"]["candidates"] = time.perf_counter() - started
    except BaseException as exc:
        payload.update(reason=type(exc).__name__, detail=first_line(exc, 300))
        _write(out_path, payload, queue)
        return
    rejection = result.get("rejected")
    if rejection is not None:
        payload.update(status="phase1_rejected", reason=rejection["reason"],
                       detail=rejection.get("detail", ""), phase1=result)
        _write(out_path, payload, queue)
        return
    if result.get("schema") != PHASE1_SCHEMA:
        payload.update(reason="schema",
                       detail=f"phase 1 wrote {result.get('schema')}, "
                              f"this harness reads {PHASE1_SCHEMA}")
        _write(out_path, payload, queue)
        return
    payload.update(status="ok", phase1=result)
    _write(out_path, payload, queue)


def _write(out_path: str, payload: dict[str, Any], queue) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    temporary.replace(path)
    queue.put({
        "status": payload["status"],
        "reason": payload.get("reason"),
        "detail": (payload.get("detail") or "")[:200],
        "seconds": payload["seconds"],
        "bytes": path.stat().st_size,
    })


def fill_one(circuit: Circuit, cache: Path = CACHE_DIR) -> dict[str, Any]:
    """Run one circuit and write its file; the summary the child reported."""
    path = file_of(circuit, cache)
    meta = {
        "label": circuit.label,
        "family": circuit.family,
        "program": circuit.program,
        "function": circuit.function,
        "kernel_path": str(circuit.kernel_path),
    }
    limits = ChildLimits(ADDRESS_SPACE_LIMIT, TIMEOUT_SECONDS)
    outcome = run_in_child(
        _worker, (str(circuit.kernel_path), circuit.function, str(path), meta), limits
    )
    if outcome.answer is not None:
        return outcome.answer
    # The child died without writing: record that as the circuit's file, so
    # that the report can count it and the criterion "no circuit panics or
    # times out" can name it.
    payload = {
        "schema": SCHEMA, **meta, "status": outcome.status,
        "reason": outcome.status, "detail": outcome.detail,
        "seconds": {}, "phase0": None, "phase1": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"status": outcome.status, "reason": outcome.status,
            "detail": outcome.detail, "seconds": {}, "bytes": path.stat().st_size}


def fill(circuits: list[Circuit], cache: Path = CACHE_DIR, force: bool = False) -> int:
    """Fill the cache for `circuits`, skipping current files unless `force`."""
    done = skipped = 0
    for circuit in circuits:
        path = file_of(circuit, cache)
        if not force and is_current(path):
            skipped += 1
            continue
        summary = fill_one(circuit, cache)
        done += 1
        seconds = summary.get("seconds") or {}
        print(
            f"  {summary['status']:<16} {circuit.label:<54} "
            f"phase0 {seconds.get('analyze', 0.0):6.1f}s  both {seconds.get('candidates', 0.0):6.1f}s  "
            f"{summary.get('bytes', 0) / 1e6:7.1f} MB"
            + (f"  {summary.get('reason', '')}: {summary.get('detail', '')[:100]}"
               if summary["status"] not in ("ok",) else ""),
            flush=True,
        )
    print(f"filled {done}, kept {skipped}, cache {cache} holds "
          f"{size_bytes(cache) / 1e6:.1f} MB", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="the whole parity set")
    parser.add_argument("--only", action="append", default=[],
                        help="only the circuits whose label contains this text; repeatable")
    parser.add_argument("--force", action="store_true",
                        help="re-run a circuit whose file is current")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    arguments = parser.parse_args()
    circuits = parity_set()
    if arguments.only:
        circuits = [c for c in circuits if any(p in c.label for p in arguments.only)]
    elif not arguments.all:
        parser.error("say --all, or --only TEXT")
    print(f"parity set: {len(circuits)} circuits; cache {arguments.cache}", flush=True)
    return fill(circuits, arguments.cache, arguments.force)


if __name__ == "__main__":
    raise SystemExit(main())
