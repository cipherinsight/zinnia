"""Running one step in its own child process, under a memory cap and a timeout.

The memory cap is the one `eval-redo/tools/memrun.sh` applies: a transient
systemd user scope with `MemoryMax` and no swap, the child marked as the
preferred OOM victim (`choom -n 1000`), so a runaway step is killed alone.
One addition: `OOMPolicy=continue`. With systemd's default (`stop`), an OOM kill
of a tool the step started (nargo, bb, a prover) makes systemd stop the whole
scope with SIGTERM before the step can report; with `continue` only the killed
process dies, and the step records `out_of_memory` for it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def memory_capped(command: list[str], memory_max: str | None) -> list[str]:
    """`command` wrapped in a memory-capped systemd scope (`None` = no cap)."""
    if memory_max is None:
        return list(command)
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
            "--", "choom", "-n", "1000", "--", *command]


@dataclass
class ChildOutcome:
    returncode: int | None
    timed_out: bool
    wall_s: float
    command: list[str]

    @property
    def killed_by_sigkill(self) -> bool:
        return self.returncode == -signal.SIGKILL

    def to_json(self) -> dict:
        return asdict(self)


def run_child(command: list[str], *, env: dict[str, str], cwd: Path, timeout_s: float,
              memory_max: str | None, log_path: Path) -> ChildOutcome:
    """Run `command`, appending its output to `log_path`; kill its session on timeout."""
    full = memory_capped(command, memory_max)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("ab") as log:
        log.write(f"$ {' '.join(full)}\n".encode())
        log.flush()
        proc = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                start_new_session=True)
        timed_out = False
        try:
            returncode: int | None = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_session(proc)
            returncode = proc.wait()
    return ChildOutcome(returncode, timed_out, time.perf_counter() - started, full)


def _kill_session(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def oom_kills_in_own_cgroup() -> int:
    """How many processes the kernel OOM-killed in this process's cgroup (0 if unknown).

    A step runs alone in its memory-capped scope, so a non-zero count means a
    process of this step (a prover, nargo, bb) hit the cap.
    """
    try:
        entry = Path("/proc/self/cgroup").read_text().strip().splitlines()[0]
        events = Path("/sys/fs/cgroup" + entry.split(":", 2)[2]) / "memory.events"
        for line in events.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == "oom_kill":
                return int(value)
    except (OSError, IndexError, ValueError):
        pass
    return 0
