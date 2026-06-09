#!/usr/bin/env python3
"""Compile-check generated DS-1000 NumPy kernels."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from validate_contracts import _compile_one_worker, function_roles

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "ds1000_numpy_compile_validation.json"


def start_job(ctx: mp.context.BaseContext, path: Path, fn_name: str) -> tuple[mp.Process, mp.Queue, float]:
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_compile_one_worker, args=(str(path), fn_name, queue))
    proc.start()
    return proc, queue, time.monotonic()


def classify_error(error_type: str | None, error_message: str | None) -> str:
    msg = (error_message or "").lower()
    typ = (error_type or "").lower()
    if "timeout" in typ:
        return "timeout"
    if "missing" in msg and "annotation" in msg:
        return "missing_type_annotation"
    if "unsupported" in msg or "not supported" in msg or "cannot compile" in msg:
        return "unsupported_language_or_operator"
    if "attribute" in msg or "has no attribute" in msg:
        return "unsupported_attribute_or_method"
    if "import" in msg or "module" in msg:
        return "import_or_runtime_dependency"
    if "assertion" in typ or "panic" in msg or "panicked" in msg:
        return "compiler_internal_error"
    if "parse" in msg or "syntax" in msg:
        return "syntax_or_frontend_parse"
    if "processexit" in typ:
        return "compiler_process_exit"
    return "other_compile_error"


def validate(output: Path, timeout_seconds: float, jobs: int) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("ds1000_case*/kernel.py")):
        for fn_name, role in function_roles(path).items():
            tasks.append(
                {
                    "program": path.parent.name,
                    "function": fn_name,
                    "kernel_path": str(path.relative_to(ROOT)),
                    "path": path,
                    "role": role,
                }
            )

    results: list[dict[str, Any]] = []
    pending = iter(tasks)
    active: list[tuple[dict[str, Any], mp.Process, mp.Queue, float]] = []
    ctx = mp.get_context("spawn")

    def enqueue_until_full() -> None:
        while len(active) < jobs:
            try:
                task = next(pending)
            except StopIteration:
                return
            rec = {k: v for k, v in task.items() if k != "path"}
            if task["role"] != "circuit":
                rec.update(status=f"{task['role']}_not_compiled", error_type=None, error_message=None)
                results.append(rec)
                continue
            proc, queue, started = start_job(ctx, task["path"], task["function"])
            active.append((rec, proc, queue, started))

    enqueue_until_full()
    while active:
        now = time.monotonic()
        next_active: list[tuple[dict[str, Any], mp.Process, mp.Queue, float]] = []
        for rec, proc, queue, started in active:
            if proc.is_alive() and now - started >= timeout_seconds:
                proc.terminate()
                proc.join(2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                rec.update(
                    status="compile_timeout",
                    error_type="Timeout",
                    error_message=f"compile exceeded {timeout_seconds:g}s timeout",
                )
                results.append(rec)
                continue
            if proc.is_alive():
                next_active.append((rec, proc, queue, started))
                continue
            proc.join()
            if not queue.empty():
                rec.update(queue.get())
            elif proc.exitcode == 0:
                rec.update(status="compile_ok", error_type=None, error_message=None)
            else:
                rec.update(
                    status="compile_error",
                    error_type="ProcessExit",
                    error_message=f"compiler process exited with code {proc.exitcode}",
                )
            results.append(rec)
        active = next_active
        enqueue_until_full()
        if active:
            time.sleep(0.2)

    results.sort(key=lambda row: (row["kernel_path"], row["function"]))
    for row in results:
        row["failure_category"] = (
            None
            if row["status"] in {"compile_ok", "chip_not_compiled", "plain_not_compiled"}
            else classify_error(row.get("error_type"), row.get("error_message"))
        )

    counts = Counter(row["status"] for row in results)
    category_counts = Counter(row["failure_category"] for row in results if row["failure_category"])
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        cat = row.get("failure_category")
        if cat and len(examples[cat]) < 8:
            examples[cat].append(
                {
                    "program": row["program"],
                    "error_type": row.get("error_type"),
                    "error_message": row.get("error_message"),
                }
            )

    payload = {
        "summary": {
            "functions": len(results),
            "circuits": sum(1 for row in results if row.get("role") == "circuit"),
            "counts": dict(counts),
            "failure_categories": dict(category_counts),
            "timeout_seconds": timeout_seconds,
            "parallel_jobs": jobs,
            "examples_by_category": dict(examples),
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    payload = validate(args.output, args.timeout_seconds, args.jobs)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
