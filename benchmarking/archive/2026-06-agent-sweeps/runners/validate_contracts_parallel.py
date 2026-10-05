#!/usr/bin/env python3
"""Parallel isolated compile-check for annotated benchmark contracts."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path
from typing import Any

from validate_contracts import DEFAULT_OUTPUT, function_roles, kernel_paths


ROOT = Path(__file__).resolve().parent


def _worker(path_text: str, fn_name: str, queue: mp.Queue) -> None:
    from validate_contracts import _compile_one_worker

    _compile_one_worker(path_text, fn_name, queue)


def _start_job(
    ctx: mp.context.BaseContext,
    path: Path,
    fn_name: str,
) -> tuple[mp.Process, mp.Queue, float]:
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(str(path), fn_name, queue))
    proc.start()
    return proc, queue, time.monotonic()


def validate_parallel(output: Path, timeout_seconds: float, jobs: int) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for path in kernel_paths():
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
                rec.update(
                    status=f"{task['role']}_not_compiled",
                    error_type=None,
                    error_message=None,
                )
                results.append(rec)
                continue
            proc, queue, started = _start_job(ctx, task["path"], task["function"])
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
            "parallel_jobs": jobs,
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
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    payload = validate_parallel(args.output, args.timeout_seconds, args.jobs)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
