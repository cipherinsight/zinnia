#!/usr/bin/env python3
"""Compile one kernel with Zinnia, and optionally mock-prove it on a fixture input,
in a child process with a time limit (and a memory cap when `MEMRUN` is set).

The child prints one JSON object:

    {"compiled": bool, "compile_error": str | null, "compile_seconds": float,
     "mock": null | {"satisfied": bool, "publics": [...], "elided": [...],
                     "error": str | null, "seconds": float},
     "backends": {"halo2-base": "ok (<cell statistics>)" | error, "noir": "ok" | error}  # with --structural}

`MEMRUN` names a wrapper such as `eval-redo/tools/memrun.sh` (it runs its
arguments under a cgroup memory limit read from `MEMRUN_MAX`). Only the mock
backend is used: it evaluates the circuit's IR on the witness and reports every
public value, which is enough to check that a fixture is accepted and that the
outputs match NumPy. Timing is the harness's job, not this tool's.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
REPO = THIS.parents[3]

NUMPY_DTYPES = {"float": np.float64, "int": np.int64, "bool": np.bool_, "complex": np.complex128}
# Backends whose circuit synthesis `--structural` checks (emit_structural: no proof is made).
BACKENDS = ("halo2-base", "noir")


def fixture_args(fixture: dict[str, Any], input_name: str) -> list[Any]:
    """Kernel arguments in signature order, as NumPy values."""
    values = fixture["inputs"][input_name]
    args = []
    for param in fixture["parameters"]:
        raw = values[param["name"]]
        dtype = NUMPY_DTYPES[param["dtype"]]
        if param["dtype"] == "complex":
            raw = np.asarray(raw, dtype=np.float64)
            raw = raw[..., 0] + 1j * raw[..., 1]
        if param["kind"] == "array":
            args.append(np.asarray(raw, dtype=dtype).reshape(param["shape"]))
        else:
            args.append(dtype(raw).item())
    return args


def load_kernel(kernel_path: Path, function: str = "kernel"):
    spec = importlib.util.spec_from_file_location(f"probe_{kernel_path.parent.name}", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, function)


def worker(kernel_path: Path, fixture_path: Path | None, input_name: str | None,
           function: str = "kernel", structural: bool = False) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from zinnia.api.zk_circuit import ZKCircuit

    result: dict[str, Any] = {"compiled": False, "compile_error": None, "compile_seconds": None, "mock": None}
    start = time.monotonic()
    try:
        compiled = ZKCircuit.from_method(load_kernel(kernel_path, function)).compile()
        result["compiled"] = True
    except BaseException as exc:  # noqa: BLE001 - pyo3 panics are BaseException
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["compile_seconds"] = round(time.monotonic() - start, 3)
    if fixture_path is None or input_name is None:
        return result
    mock: dict[str, Any] = {"satisfied": False, "publics": [], "elided": [], "error": None}
    start = time.monotonic()
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        proof = compiled.prove(*fixture_args(fixture, input_name), backend="mock")
        mock["satisfied"] = proof.proof_bytes_hex == "mock_satisfied"
        mock["publics"] = list(proof.public_values)
        mock["elided"] = list(compiled.get_elided_publics())
    except BaseException as exc:  # noqa: BLE001
        mock["error"] = f"{type(exc).__name__}: {exc}"
        mock["traceback"] = traceback.format_exc(limit=4)
    mock["seconds"] = round(time.monotonic() - start, 3)
    result["mock"] = mock
    if structural and mock["error"] is None:
        result["backends"] = {}
        for backend in BACKENDS:
            try:
                text = compiled.emit_structural(*fixture_args(fixture, input_name), backend=backend)
                result["backends"][backend] = "ok" + (f" ({text})" if backend == "halo2-base" else "")
            except BaseException as exc:  # noqa: BLE001
                result["backends"][backend] = f"{type(exc).__name__}: {exc}"
    return result


def run(kernel_path: Path, fixture_path: Path | None = None, input_name: str | None = None,
        timeout: float = 300.0, function: str = "kernel", structural: bool = False) -> dict[str, Any]:
    """Run the worker in a child process; a timeout or crash becomes an error."""
    command = [sys.executable, str(THIS), "--worker", str(kernel_path), "--function", function]
    if structural:
        command.append("--structural")
    if fixture_path is not None:
        command += ["--fixture", str(fixture_path), "--input", str(input_name)]
    wrapper = os.environ.get("MEMRUN")
    if wrapper:
        command = [wrapper, *command]
    start = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired:
        return {"compiled": False, "compile_error": f"timeout after {timeout:.0f} s", "compile_seconds": None,
                "mock": None}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        seconds = time.monotonic() - start
        if proc.returncode == -9 and wrapper:
            message = (f"killed by signal 9 after {seconds:.0f} s: exceeded the memory cap "
                       f"(MEMRUN_MAX={os.environ.get('MEMRUN_MAX', '8G')})")
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            message = f"child exited {proc.returncode} after {seconds:.0f} s: {' | '.join(tail)}"
        return {"compiled": False, "compile_error": message, "compile_seconds": None, "mock": None}
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--input", choices=["A", "B"])
    parser.add_argument("--function", default="kernel", help="name of the @zk_circuit function")
    parser.add_argument("--structural", action="store_true",
                        help="with --fixture: also synthesize the circuit for each backend in BACKENDS")
    parser.add_argument("--worker", action="store_true", help="run in this process (used by run())")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.kernel.resolve(), args.fixture, args.input, args.function, args.structural)))
    else:
        print(json.dumps(run(args.kernel.resolve(), args.fixture, args.input, args.timeout, args.function,
                             args.structural), indent=2))


if __name__ == "__main__":
    main()
