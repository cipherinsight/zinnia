#!/usr/bin/env python3
"""Measure one (program, backend) under the compiler configuration in os.environ.

Run by run.py with the checkout's interpreter, cwd = the checkout:

    python child.py CHECKOUT KERNEL FIXTURE FUNCTION BACKEND

Prints one JSON line. Nothing is proved: halo2-base sizes come from the witness-free
`halo2_base_cell_stats`, Noir sizes from `nargo compile` + `bb gates` of the emitted
`main.nr` (input B), exactly as benchmarking/ablations/common.py measures them.

Size definitions:
  halo2: size = advice + lookup cells of `halo2_base_cell_stats` at the k estimate_params
         picks (advice includes second-phase, lookup-table and RAM-trace advice cells; the
         earlier ablations used the same sum). NOTE: RQ2 (fig/data/rq2-perf.py `size`) parses
         the harness fingerprint, whose advice is the base builder's only: it leaves out
         table and RAM-trace advice, so it is smaller wherever a table read or a RAM trace
         is chosen (ds1000_case0319: 14,833 vs 15,984).
  noir:  size = bb gates circuit_size (identical to RQ2)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha16(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()[:16]


def measure(checkout: Path, kernel: str, fixture: str, function: str, backend: str) -> dict:
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(checkout / "benchmarking"))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    started = time.perf_counter()
    compiled = compile_kernel(Path(kernel), function, backend=backend)
    record: dict = {"compile_s": round(time.perf_counter() - started, 3)}
    stmts = compiled.get_ir_stmts()
    record.update(ir=sha16(json.dumps(stmts, sort_keys=True)), stmts=len(stmts))
    if backend == "halo2":
        from zinnia._zinnia_core import halo2_base_cell_stats
        stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
        record.update({k: stats.get(k) for k in (
            "k", "lookup_bits", "advice", "lookup", "second_advice", "fixed", "challenges",
            "committed_polys", "poly_cost")})
        # cell_stats' `advice` already includes the second-phase advice and the advice
        # cells of lookup tables and RAM traces (src/prove/halo2_base/mod.rs cell_stats).
        record["size"] = stats["advice"] + stats["lookup"]
    else:
        source = compiled.emit_structural(*load_fixture(Path(fixture)).arguments("B"),
                                          backend="noir")
        record["noir"] = sha16(source)
        from noir_gates import NOIR_FPC, gate_count
        started = time.perf_counter()
        gates = gate_count(source, "noir_fpc", NOIR_FPC)
        record["gates_s"] = round(time.perf_counter() - started, 3)
        if "error" in gates:
            return {**record, "error": gates["error"]}
        record.update(gates)
        record["size"] = gates["gates"]
    return record


def main(argv: list[str]) -> int:
    checkout, kernel, fixture, function, backend = argv
    try:
        record = measure(Path(checkout), kernel, fixture, function, backend)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, KeyboardInterrupt):
            raise
        record = {"error": f"{type(exc).__name__}: {exc}"[:1500]}
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
