"""Run one command under a memory cap and a timeout, its output passed through.

The remote container has no systemd and a read-only cgroup tree, so `memrun.sh` cannot
work there. This uses the same mechanism as the harness's timing epochs
(`ZINNIA_HARNESS_MEMORY_CAP=watchdog`, `benchmarking/harness/isolation.py`): the command
runs in its own session, the summed RSS of the session is sampled every 0.1 s, and the
whole session is SIGKILLed above the cap.

Usage: python capped.py MEMORY TIMEOUT_S CMD [ARGS...]
  MEMORY like 30G ('none' = no cap); TIMEOUT_S in seconds ('none' = no timeout).
Exit status: the command's; 137 when killed over the cap (a line on stderr says so),
124 on timeout. The peak session RSS is printed on stderr as `capped: peak_rss_bytes=N`.
`ZWT` (or the current directory) must be a Zinnia checkout, for the harness module.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get("ZWT", os.getcwd()))
from benchmarking.harness.isolation import (  # noqa: E402
    WATCHDOG_INTERVAL_S, _kill_session, _session_members, size_in_bytes)


def main() -> int:
    memory, timeout, command = sys.argv[1], sys.argv[2], sys.argv[3:]
    limit = None if memory.lower() == "none" else size_in_bytes(memory)
    deadline = None if timeout.lower() == "none" else time.monotonic() + float(timeout)
    proc = subprocess.Popen(["choom", "-n", "1000", "--", *command], start_new_session=True)
    peak = 0
    status = None
    while status is None:
        try:
            status = proc.wait(timeout=WATCHDOG_INTERVAL_S)
            break
        except subprocess.TimeoutExpired:
            pass
        rss = sum(r for _, r in _session_members(proc.pid))
        peak = max(peak, rss)
        if limit is not None and rss > limit:
            print(f"capped: over the {memory} cap ({rss} bytes RSS); killed", file=sys.stderr, flush=True)
            _kill_session(proc)
            proc.wait()
            status = 137
        elif deadline is not None and time.monotonic() > deadline:
            print(f"capped: timeout after {timeout} s; killed", file=sys.stderr, flush=True)
            _kill_session(proc)
            proc.wait()
            status = 124
    print(f"capped: peak_rss_bytes={peak}", file=sys.stderr, flush=True)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
