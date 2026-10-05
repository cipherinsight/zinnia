#!/usr/bin/env python3
"""Compile-check annotated benchmark contracts and summarize outcomes."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import multiprocessing as mp
import sys
import traceback
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "compile_validation.json"
DEFAULT_TIMEOUT_SECONDS = 20


def load_module(path: Path) -> ModuleType:
    module_name = "zinnia_benchmark_" + path.parent.name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def function_roles(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    roles: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = {ast.unparse(dec) for dec in node.decorator_list}
        if "zk_circuit" in decorators:
            roles[node.name] = "circuit"
        elif "zk_chip" in decorators:
            roles[node.name] = "chip"
        else:
            roles[node.name] = "plain"
    return roles


def kernel_paths() -> list[Path]:
    """Every kernel of the three dataset families, in a stable order."""
    families = ("ds1000", "npbench", "pythran")
    return sorted(
        path
        for family in families
        for path in (ROOT / "datasets" / family).glob("*/kernel.py")
    )


def validate(output: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from zinnia import ZKCircuit

    results: list[dict[str, Any]] = []
    for path in kernel_paths():
        try:
            module = load_module(path)
            module_error = None
        except Exception as exc:  # import-time unsupported helper/defaults
            module = None
            module_error = exc
        for fn_name, role in function_roles(path).items():
            rec = {
                "program": path.parent.name,
                "function": fn_name,
                "kernel_path": str(path.relative_to(ROOT)),
                "role": role,
                "status": "unknown",
                "error_type": None,
                "error_message": None,
            }
            if role != "circuit":
                rec["status"] = f"{role}_not_compiled"
                results.append(rec)
                continue
            if module_error is not None:
                rec.update(
                    status="import_error",
                    error_type=type(module_error).__name__,
                    error_message=str(module_error).splitlines()[0][:500],
                )
                results.append(rec)
                continue
            fn = getattr(module, fn_name, None)
            if fn is None:
                rec.update(status="missing_function", error_message="function not found after import")
                results.append(rec)
                continue
            try:
                ZKCircuit.from_method(fn).compile()
                rec["status"] = "compile_ok"
            except Exception as exc:
                rec.update(
                    status="compile_error",
                    error_type=type(exc).__name__,
                    error_message=(str(exc).splitlines()[0] if str(exc) else repr(exc))[:500],
                )
            results.append(rec)

    counts = Counter(row["status"] for row in results)
    circuit_counts = Counter(row["status"] for row in results if row.get("role") == "circuit")
    by_error = Counter(
        (row["error_type"] or "", row["error_message"] or "")
        for row in results
        if row["status"] != "compile_ok"
    )
    payload = {
        "summary": {
            "functions": len(results),
            "circuits": sum(1 for row in results if row.get("role") == "circuit"),
            "counts": dict(counts),
            "circuit_counts": dict(circuit_counts),
            "top_errors": [
                {"count": count, "error_type": key[0], "error_message": key[1]}
                for key, count in by_error.most_common(40)
            ],
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _compile_one_worker(path_text: str, fn_name: str, queue) -> None:
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from zinnia import ZKCircuit

        module = load_module(Path(path_text))
        fn = getattr(module, fn_name)
        ZKCircuit.from_method(fn).compile()
        queue.put({"status": "compile_ok", "error_type": None, "error_message": None})
    except BaseException as exc:
        queue.put(
            {
                "status": "compile_error",
                "error_type": type(exc).__name__,
                "error_message": (str(exc).splitlines()[0] if str(exc) else repr(exc))[:500],
                "traceback_tail": traceback.format_exc(limit=4).splitlines()[-8:],
            }
        )


def compile_one(path: Path, fn_name: str, timeout_seconds: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_compile_one_worker, args=(str(path), fn_name, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            "status": "compile_timeout",
            "error_type": "Timeout",
            "error_message": f"compile exceeded {timeout_seconds:g}s timeout",
        }
    if not queue.empty():
        return queue.get()
    if proc.exitcode == 0:
        return {"status": "compile_ok", "error_type": None, "error_message": None}
    return {
        "status": "compile_error",
        "error_type": "ProcessExit",
        "error_message": f"compiler process exited with code {proc.exitcode}",
    }


def validate_isolated(output: Path, timeout_seconds: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for path in kernel_paths():
        roles = function_roles(path)
        for fn_name, role in roles.items():
            rec = {
                "program": path.parent.name,
                "function": fn_name,
                "kernel_path": str(path.relative_to(ROOT)),
                "role": role,
                "status": "unknown",
                "error_type": None,
                "error_message": None,
            }
            if role != "circuit":
                rec["status"] = f"{role}_not_compiled"
            else:
                rec.update(compile_one(path, fn_name, timeout_seconds))
            results.append(rec)

    counts = Counter(row["status"] for row in results)
    circuit_counts = Counter(row["status"] for row in results if row.get("role") == "circuit")
    by_error = Counter(
        (row["error_type"] or "", row["error_message"] or "")
        for row in results
        if row["status"] not in {"compile_ok", "chip_not_compiled", "plain_not_compiled"}
    )
    payload = {
        "summary": {
            "functions": len(results),
            "circuits": sum(1 for row in results if row.get("role") == "circuit"),
            "counts": dict(counts),
            "circuit_counts": dict(circuit_counts),
            "timeout_seconds": timeout_seconds,
            "top_errors": [
                {"count": count, "error_type": key[0], "error_message": key[1]}
                for key, count in by_error.most_common(40)
            ],
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run validation in the current process. Faster, but a slow/panicking kernel can stall the full pass.",
    )
    args = parser.parse_args()
    if args.in_process:
        payload = validate(args.output)
    else:
        payload = validate_isolated(args.output, args.timeout_seconds)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
