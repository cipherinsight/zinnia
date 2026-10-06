"""Running one step in its own child process, under a memory cap and a timeout.

The memory cap is enforced in one of two ways, chosen by the environment variable
`ZINNIA_HARNESS_MEMORY_CAP` (recorded in the manifest):

- `systemd` (the default): the cap `eval-redo/tools/memrun.sh` applies, a
  transient systemd user scope with `MemoryMax` and no swap, the child marked as
  the preferred OOM victim (`choom -n 1000`), so a runaway step is killed alone.
  One addition: `OOMPolicy=continue`. With systemd's default (`stop`), an OOM kill
  of a tool the step started (nargo, bb, a prover) makes systemd stop the whole
  scope with SIGTERM before the step can report; with `continue` only the killed
  process dies, and the step records `out_of_memory` for it.
- `watchdog`: for machines without a systemd user manager or a writable cgroup
  tree (a Docker container). The harness samples the resident memory (RSS) of
  every process in the step's session every `WATCHDOG_INTERVAL_S` and kills the
  whole session when the sum exceeds the cap; the step is then `out_of_memory`.
  It differs from the cgroup cap in two ways: it counts RSS (shared pages once per
  process, no page cache), and a peak shorter than one interval can pass unseen.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

#: Selects how the memory cap is enforced: "systemd" (default) or "watchdog".
CAP_VARIABLE = "ZINNIA_HARNESS_MEMORY_CAP"
CAP_MECHANISMS = ("systemd", "watchdog")
#: How often the watchdog samples the step's memory.
WATCHDOG_INTERVAL_S = 0.1

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def cap_mechanism() -> str:
    """The memory-cap mechanism this process uses (`ZINNIA_HARNESS_MEMORY_CAP`)."""
    mechanism = os.environ.get(CAP_VARIABLE, "systemd")
    if mechanism not in CAP_MECHANISMS:
        raise ValueError(f"{CAP_VARIABLE}={mechanism!r}; expected one of {CAP_MECHANISMS}")
    return mechanism


def size_in_bytes(text: str) -> int:
    """A systemd-style size (`16G`, `512M`, `1048576`; 1024-based units) in bytes."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"not a memory size: {text!r}")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()])


def memory_capped(command: list[str], memory_max: str | None) -> list[str]:
    """`command` as the step's child is started (`None` = no cap).

    With the systemd mechanism the command runs in a memory-capped scope; with the
    watchdog it is only made the preferred OOM victim, and `run_child` watches it.
    """
    if memory_max is None:
        return list(command)
    victim = ["choom", "-n", "1000", "--", *command]
    if cap_mechanism() == "watchdog":
        return victim
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0", "-p", "OOMPolicy=continue",
            "--", *victim]


def cap_problem(memory_max: str | None) -> str | None:
    """Why the configured memory cap cannot be applied here (`None` if it can).

    Without this check, a machine whose systemd user manager is unreachable would
    report every step as `crashed`.
    """
    if memory_max is None:
        return None
    try:
        size_in_bytes(memory_max)
        if cap_mechanism() == "watchdog":
            return None
        proc = subprocess.run(memory_capped(["true"], "64M"), capture_output=True, text=True,
                              timeout=60)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        return (f"a memory-capped systemd scope does not start ({detail}); on a machine "
                f"without a systemd user manager set {CAP_VARIABLE}=watchdog")
    return None


@dataclass
class ChildOutcome:
    returncode: int | None
    timed_out: bool
    wall_s: float
    command: list[str]
    #: Watchdog only: the largest session RSS seen, and whether it passed the cap.
    peak_rss_bytes: int | None = None
    over_memory_cap: bool = False

    @property
    def killed_by_sigkill(self) -> bool:
        return self.returncode == -signal.SIGKILL

    def to_json(self) -> dict:
        return asdict(self)


def run_child(command: list[str], *, env: dict[str, str], cwd: Path, timeout_s: float,
              memory_max: str | None, log_path: Path) -> ChildOutcome:
    """Run `command`, appending its output to `log_path`; kill its session on timeout
    (and, with the watchdog, when its memory passes the cap)."""
    full = memory_capped(command, memory_max)
    limit = (size_in_bytes(memory_max)
             if memory_max is not None and cap_mechanism() == "watchdog" else None)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + timeout_s
    with log_path.open("ab") as log:
        log.write(f"$ {' '.join(full)}\n".encode())
        log.flush()
        proc = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                start_new_session=True)
        timed_out = over_cap = False
        peak = 0 if limit is not None else None
        while True:
            wait_s = deadline - time.perf_counter()
            if limit is not None:
                wait_s = min(wait_s, WATCHDOG_INTERVAL_S)
            try:
                returncode: int | None = proc.wait(timeout=max(wait_s, 0.0))
                break
            except subprocess.TimeoutExpired:
                pass
            if time.perf_counter() >= deadline:
                timed_out = True
            elif limit is not None:
                rss = sum(member_rss for _, member_rss in _session_members(proc.pid))
                peak = max(peak, rss)
                over_cap = rss > limit
                if over_cap:
                    log.write(f"\nharness: the step's processes use {rss} bytes of RSS, over the "
                              f"{memory_max} cap; killing them\n".encode())
            if timed_out or over_cap:
                _kill_session(proc)
                returncode = proc.wait()
                break
    return ChildOutcome(returncode, timed_out, time.perf_counter() - started, full, peak, over_cap)


def _session_members(session: int) -> list[tuple[int, int]]:
    """(pid, RSS bytes) of every live process whose session id is `session`."""
    members = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path("/proc", entry.name, "stat").read_text()
        except OSError:  # the process exited meanwhile
            continue
        # Fields after the command name (which may contain spaces): state is field 3,
        # session field 6, rss (pages) field 24.
        fields = stat[stat.rindex(")") + 2:].split()
        if int(fields[3]) == session:
            members.append((int(entry.name), int(fields[21]) * PAGE_SIZE))
    return members


def _kill_session(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group and any process left in its session."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, _ in _session_members(proc.pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def oom_kills_in_own_cgroup() -> int:
    """How many processes the kernel OOM-killed in this process's cgroup (0 if unknown).

    A step runs alone in its memory-capped scope, so a non-zero count means a
    process of this step (a prover, nargo, bb) hit the cap. Under the watchdog the
    step has no cgroup of its own (its cgroup may be a whole container's), so the
    count says nothing about the step and is not read.
    """
    if cap_mechanism() == "watchdog":
        return 0
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
