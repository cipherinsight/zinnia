"""Running one step in its own child process, under a memory cap and a timeout.

The memory cap is the one `eval-redo/tools/memrun.sh` applies: a transient
systemd user scope with `MemoryMax` and no swap, the child marked as the
preferred OOM victim (`choom -n 1000`), so a runaway step is killed alone.
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
            "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0",
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
