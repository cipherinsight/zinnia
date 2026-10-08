"""Watch a zkVM prover while it runs: how many threads it has and how many CPUs it keeps busy.

    python -m benchmarking.scaling_npb.thread_probe --out FILE -- <command ...>

Runs the command (typically a harness run of one case on `sp1` or `risc0` with
`--threads N`) and samples /proc every 0.5 s. For every process whose name is
a zkVM host (`sp1-baseline`, `risc0-baseline`) or a process it started, it
records the thread count (`Threads:` in /proc/PID/status), the thread
variables in its environment (RAYON_NUM_THREADS, OMP_NUM_THREADS), its CPU
affinity, and the CPU time it used per sample interval. The summary gives,
per host process, the largest thread count and the busiest interval in CPUs
(CPU seconds per wall second), so one can see that `--threads` reaches the
prover and how much of the machine it uses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOSTS = ("sp1-baseline", "risc0-baseline")
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
INTERVAL_S = 0.5


def processes() -> dict[int, tuple[str, int, int]]:
    """pid -> (name, parent pid, CPU ticks used so far) for every live process."""
    found = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path("/proc", entry.name, "stat").read_text()
        except OSError:
            continue
        name = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        found[int(entry.name)] = (name, int(fields[1]), int(fields[11]) + int(fields[12]))
    return found


def describe(pid: int) -> dict:
    record: dict = {}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        record["threads"] = int(next(l for l in status.splitlines() if l.startswith("Threads:")).split()[1])
        record["cpus_allowed"] = next(l for l in status.splitlines()
                                      if l.startswith("Cpus_allowed_list:")).split()[1]
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        record["env"] = {k: v for k, _, v in (e.decode(errors="replace").partition("=") for e in environ)
                         if k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "HARDWARE_CONCURRENCY")}
    except (OSError, StopIteration):
        pass
    return record


def watched(table: dict[int, tuple[str, int, int]]) -> dict[int, int]:
    """Host processes and their descendants: pid -> the host pid they belong to."""
    roots = {pid for pid, (name, _, _) in table.items() if name in HOSTS}
    owner = {pid: pid for pid in roots}
    changed = True
    while changed:
        changed = False
        for pid, (_, parent, _) in table.items():
            if pid not in owner and parent in owner:
                owner[pid] = owner[parent]
                changed = True
    return owner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    proc = subprocess.Popen(command)
    hosts: dict[int, dict] = {}
    previous: dict[int, int] = {}
    last = time.monotonic()
    while proc.poll() is None:
        time.sleep(INTERVAL_S)
        now = time.monotonic()
        table = processes()
        owner = watched(table)
        busy: dict[int, int] = {}
        for pid, host in owner.items():
            ticks = table[pid][2]
            busy[host] = busy.get(host, 0) + ticks - previous.get(pid, ticks)
            previous[pid] = ticks
        for host, ticks in busy.items():
            record = hosts.setdefault(host, {"name": table[host][0], "max_threads": 0, "max_busy_cpus": 0.0,
                                             "samples": 0})
            info = describe(host)
            record["max_threads"] = max(record["max_threads"], info.get("threads", 0))
            record.update({k: info[k] for k in ("cpus_allowed", "env") if k in info})
            record["max_busy_cpus"] = max(record["max_busy_cpus"], round(ticks / CLOCK_TICKS / (now - last), 1))
            record["samples"] += 1
        last = now
    summary = {"command": command, "exit": proc.returncode, "host_cpus": os.cpu_count(),
               "hosts": list(hosts.values())}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
