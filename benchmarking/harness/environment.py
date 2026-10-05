"""Machine facts, per-case environment snapshots, and the child environment.

Every system runs CPU-only (author's decision): child processes see no CUDA
device (`CUDA_VISIBLE_DEVICES=""`), SP1 is told to use its CPU prover, and
RISC Zero's dev mode (which skips proving) must not be set.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

#: Environment variables recorded in the manifest (prefixes and exact names).
TRACKED_PREFIXES = ("ZINNIA_", "SP1_", "RISC0_")
TRACKED_NAMES = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "HARDWARE_CONCURRENCY",
                 "CUDA_VISIBLE_DEVICES", "RUSTFLAGS", "TMPDIR")

#: Thread-count variables set from `--threads`: rayon (halo2, SP1, RISC Zero),
#: OpenMP, and Barretenberg's `HARDWARE_CONCURRENCY`.
THREAD_VARIABLES = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "HARDWARE_CONCURRENCY")

CPU0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _command_output(*args: str) -> str | None:
    try:
        proc = subprocess.run(list(args), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or proc.stderr).strip() or None


def _cpu_model() -> str | None:
    text = _read(Path("/proc/cpuinfo")) or ""
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1) if match else platform.processor() or None


def _memory_total_kib() -> int | None:
    text = _read(Path("/proc/meminfo")) or ""
    match = re.search(r"^MemTotal:\s*(\d+) kB", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def machine_info() -> dict:
    """Static facts about the machine, recorded once per epoch."""
    return {
        "hostname": platform.node(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "lscpu_topology": _command_output("sh", "-c",
                                          "lscpu | grep -E '^(Thread|Core|Socket)'"),
        "memory_total_kib": _memory_total_kib(),
        "kernel": platform.release(),
        "glibc": " ".join(platform.libc_ver()),
        "gpu": _command_output("nvidia-smi", "-L"),
    }


def cpu_state() -> dict:
    """Frequency policy and load, recorded before every case."""
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loadavg": list(os.getloadavg()),
        "governor": _read(CPU0 / "scaling_governor"),
        "energy_performance_preference": _read(CPU0 / "energy_performance_preference"),
        "boost": _read(Path("/sys/devices/system/cpu/cpufreq/boost")),
        "scaling_driver": _read(CPU0 / "scaling_driver"),
    }


def tracked_variables(env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)
    return {k: v for k, v in sorted(env.items())
            if k.startswith(TRACKED_PREFIXES) or k in TRACKED_NAMES}


class UnsafeEnvironment(RuntimeError):
    """The environment would make the measurement invalid."""


def assert_cpu_only_allowed(env: dict[str, str] | None = None) -> None:
    """Refuse to run when RISC Zero's dev mode (fake proofs) is switched on."""
    env = dict(os.environ if env is None else env)
    dev_mode = env.get("RISC0_DEV_MODE", "").strip().lower()
    if dev_mode not in ("", "0", "false", "no"):
        raise UnsafeEnvironment(
            f"RISC0_DEV_MODE={env['RISC0_DEV_MODE']!r}: dev mode skips proving; unset it")


def child_environment(threads: int | None) -> dict[str, str]:
    """The environment of every measured child process."""
    assert_cpu_only_allowed()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["SP1_PROVER"] = "cpu"
    env.pop("RISC0_DEV_MODE", None)
    if threads is not None:
        for name in THREAD_VARIABLES:
            env[name] = str(threads)
    return env


def _indexer_command() -> str | None:
    for name in ("balooctl6", "balooctl"):
        found = shutil.which(name)
        if found:
            return found
    return None


@contextlib.contextmanager
def indexer_paused(log: list[str]) -> Iterator[None]:
    """Suspend KDE's file indexer for the duration of a run, if it is running.

    It is resumed afterwards only if this context suspended it. What happened
    is appended to `log` (recorded in the manifest).
    """
    command = _indexer_command()
    if command is None:
        log.append("no balooctl found; nothing paused")
        yield
        return
    status = _command_output(command, "status") or ""
    running = "is running" in status.splitlines()[0] if status else False
    if not running:
        log.append(f"{command}: indexer not running ({status.splitlines()[0] if status else '?'})")
        yield
        return
    subprocess.run([command, "suspend"], capture_output=True, timeout=60)
    log.append(f"{command} suspend at {time.strftime('%H:%M:%S')}")
    try:
        yield
    finally:
        subprocess.run([command, "resume"], capture_output=True, timeout=60)
        log.append(f"{command} resume at {time.strftime('%H:%M:%S')}")
