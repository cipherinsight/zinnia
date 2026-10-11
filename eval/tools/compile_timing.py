#!/usr/bin/env python3
"""Compile-time cost of the fact layer: compile every program under several configurations.

Programs: the DS-1000 perf set (status `perf`) and every in-scope acceptance entry function.
Configurations (environment variables; see docs/ablations.md in the compiler):
  default     nothing set
  no_smt      ZINNIA_ABLATE_SMT=1                      (no Z3 anywhere; intervals stay)
  no_facts    ZINNIA_ABLATE_SMT=1 ZINNIA_SMT_ENABLE=0  (no Z3, and the IR generator's resolver
                                                         is constants-only: close to "no facts
                                                         for bounding"; the passes keep intervals)
Each compile runs in its own process (fresh hash seeds, no shared caches) with a memory cap and a
timeout, and records: outcome, wall-clock compile time, the per-phase trace
(ZINNIA_COMPILE_PHASE_TRACE=1) and the resolver's SMT/interval time
(ZINNIA_SMT_LOG_TELEMETRY=1). Target: halo2 (the halo2-base compile target).

Usage: compile_timing.py --repo <zinnia worktree> --out results.jsonl [--configs a,b] [--jobs N]
       [--timeout S] [--only perf|acceptance]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "default": {},
    "no_smt": {"ZINNIA_ABLATE_SMT": "1"},
    "no_facts": {"ZINNIA_ABLATE_SMT": "1", "ZINNIA_SMT_ENABLE": "0"},
}
MEMRUN = Path("/home/zhantong/Projects/zinnia-project/zinnia-src/eval/tools/memrun.sh")

CHILD = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from benchmarking.harness.kernels import compile_kernel
t = time.perf_counter()
try:
    compile_kernel(Path(sys.argv[1]), sys.argv[2], backend="halo2")
    out = {"ok": True}
except BaseException as exc:
    out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
out["compile_s"] = time.perf_counter() - t
print("@@RESULT@@" + json.dumps(out), flush=True)
"""

PHASE = re.compile(r"\[zinnia-compile-phase\] phase=(\S+) elapsed_s=([0-9.]+)")
MS = re.compile(r"(total_time_in_(?:smt|range)_ms)\s*=\s*([0-9.]+)")


def programs(repo: Path, only: str | None) -> list[dict]:
    sys.path.insert(0, str(repo))
    from benchmarking.harness.case_sets import load_acceptance_programs
    out = []
    if only in (None, "perf"):
        raw = json.loads((repo / "benchmarking/datasets/case_sets/ds1000_perf.json").read_text())
        for c in raw["cases"]:
            if c["status"] == "perf":
                out.append({"set": "perf", "name": c["case"], "kernel":
                            str(repo / "benchmarking/datasets/ds1000" / c["case"] / "kernel.py"), "function": "kernel"})
    if only in (None, "acceptance"):
        for p in load_acceptance_programs(repo / "benchmarking/datasets/case_sets/acceptance.json"):
            out.append({"set": f"acceptance:{p.dataset}", "name": f"{p.program}::{p.function}",
                        "kernel": str(p.kernel_path), "function": p.function})
    return out


def run_one(repo: Path, prog: dict, config: str, timeout: float, mem: str) -> dict:
    env = dict(os.environ, ZINNIA_COMPILE_PHASE_TRACE="1", ZINNIA_SMT_LOG_TELEMETRY="1", MEMRUN_MAX=mem,
               PYTHONHASHSEED="0")
    for k in ("ZINNIA_ABLATE_SMT", "ZINNIA_SMT_ENABLE"):
        env.pop(k, None)
    env.update(CONFIGS[config])
    cmd = [str(MEMRUN), str(repo / ".venv/bin/python"), "-c", CHILD, prog["kernel"], prog["function"], str(repo)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=repo, env=env)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rc = "timeout"
    rec = {**prog, "config": config, "wall_s": time.perf_counter() - started, "rc": rc}
    m = [l for l in stdout.splitlines() if l.startswith("@@RESULT@@")]
    if m:
        rec.update(json.loads(m[-1][len("@@RESULT@@"):]))
    else:
        rec.update({"ok": False, "error": "timeout" if rc == "timeout" else f"no result (rc {rc})",
                    "compile_s": None})
    phases: dict[str, float] = {}
    for name, secs in PHASE.findall(stderr):
        phases[name] = phases.get(name, 0.0) + float(secs)
    rec["phases"] = phases
    ms: dict[str, float] = {}
    for key, val in MS.findall(stderr):  # one summary per resolver phase; summed
        ms[key] = ms.get(key, 0.0) + float(val)
    # Only the IR generator's resolver reports these; prove() and the passes' Z3 use are not
    # included (their cost shows in the difference between configurations).
    rec["resolver_smt_s"] = ms.get("total_time_in_smt_ms", 0.0) / 1e3
    rec["resolver_range_s"] = ms.get("total_time_in_range_ms", 0.0) / 1e3
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", default="default,no_smt,no_facts")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--only", choices=["perf", "acceptance"])
    args = ap.parse_args()
    progs = programs(args.repo.resolve(), args.only)
    configs = args.configs.split(",")
    jobs = [(p, c) for p in progs for c in configs]  # interleaved: the configs of one program run together
    print(f"{len(progs)} programs x {len(configs)} configs = {len(jobs)} compiles", file=sys.stderr)
    with args.out.open("w") as fh, cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(run_one, args.repo.resolve(), p, c, args.timeout, args.mem) for p, c in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            rec = f.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
