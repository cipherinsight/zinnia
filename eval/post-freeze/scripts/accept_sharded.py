"""Run `python -m benchmarking.harness accept` in parallel shards, each pinned to its own CPU, and merge.

Port of eval-redo/fix/fact-review-data/accept_sharded.py with the CPU pinning added.
Usage (cwd = a Zinnia checkout): .venv/bin/python accept_sharded.py OUT_DIR CPUS
  CPUS: comma-separated CPU ids or ranges (e.g. 0-15); one shard per CPU, programs split
  round-robin over the sorted program list. The compiler's ablation switches come from the
  environment. Harness defaults: 300 s timeout and 8G memory cap per program.
"""
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from benchmarking.harness.case_sets import load_acceptance_programs  # noqa: E402
from benchmarking.harness import paths  # noqa: E402


def cpu_list(spec: str) -> list[str]:
    cpus = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus += [str(c) for c in range(int(lo), int(hi or lo) + 1)]
    return cpus


def main(out_dir: str, cpus_spec: str) -> None:
    out, cpus = Path(out_dir), cpu_list(cpus_spec)
    out.mkdir(parents=True, exist_ok=True)
    programs = sorted({p.program for p in load_acceptance_programs(paths.ACCEPTANCE_CASE_SET)})
    k = len(cpus)
    shards = [programs[i::k] for i in range(k)]
    procs = []
    for i, shard in enumerate(shards):
        log = open(out / f"shard{i}.log", "w")
        procs.append(subprocess.Popen(
            ["taskset", "-c", cpus[i], sys.executable, "-m", "benchmarking.harness", "accept",
             "--epoch", str(out / f"shard{i}"), "--programs", ",".join(shard)],
            stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    rows = []
    for i in range(k):
        rows += json.loads((out / f"shard{i}" / "acceptance" / "results.json").read_text())["results"]
    (out / "results.json").write_text(json.dumps({"exit_codes": codes, "cpus": cpus, "results": rows}, indent=1))
    table = collections.Counter((r["dataset"], r["outcome"]) for r in rows)
    for key in sorted(table):
        print(*key, table[key])
    print("total", len(rows))


main(*sys.argv[1:])
