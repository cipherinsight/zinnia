"""The pinned toolchains of the evaluation and the `check-env` probe.

The expected versions and paths are the author's decisions of 2026-10-06
(`eval-redo/BRIEF.md`, "Decisions by the author"). Each tool lists the systems
that need it, so a run of a subset of systems only fails on the tools it uses.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarking.harness import paths

HOME = Path.home()

NOIR_SYSTEMS = ("zinnia_noir", "noir_hw")
HALO2_SYSTEMS = ("zinnia_halo2", "halo2_hw")
SP1_SYSTEMS = ("sp1", "sp1_plonk")


@dataclass(frozen=True)
class Tool:
    """One external tool: where it must be, and what its version must say."""

    name: str
    command: tuple[str, ...]
    #: A substring the version output must contain; `None` = record only.
    expected: str | None
    systems: tuple[str, ...]


TOOLS: tuple[Tool, ...] = (
    Tool("nargo", (str(HOME / ".nargo/bin/nargo"), "--version"),
         "nargo version = 1.0.0-beta.20", NOIR_SYSTEMS),
    Tool("bb", (str(HOME / ".bb/bb"), "--version"), "5.0.0-nightly.20260324", NOIR_SYSTEMS),
    # `cargo prove --version` prints a commit, not the release number; the SDK
    # version is checked in the SP1 program's Cargo.lock (see LOCKED_CRATES).
    Tool("cargo-prove", (str(HOME / ".sp1/bin/cargo-prove"), "prove", "--version"), None,
         SP1_SYSTEMS),
    Tool("rustc (succinct)", ("rustup", "run", "succinct", "rustc", "--version"), None,
         SP1_SYSTEMS),
    Tool("r0vm", (str(HOME / ".cargo/bin/r0vm"), "--version"), "risc0-r0vm 3.0.3", ("risc0",)),
    Tool("cargo-risczero", (str(HOME / ".cargo/bin/cargo-risczero"), "risczero", "--version"),
         "cargo-risczero 3.0.3", ("risc0",)),
    Tool("rustc (risc0)", ("rustup", "run", "risc0", "rustc", "--version"), None, ("risc0",)),
    Tool("scarb",
         (str(HOME / ".local/share/scarb/scarb-v2.20.1-x86_64-unknown-linux-gnu/bin/scarb"),
          "--version"),
         "scarb 2.20.1", ("cairo",)),
    Tool("rustc (default)", ("rustc", "--version"), None, HALO2_SYSTEMS),
    Tool("cargo (default)", ("cargo", "--version"), None, HALO2_SYSTEMS),
    Tool("systemd-run", ("systemd-run", "--version"), None, ()),
)


@dataclass(frozen=True)
class LockedCrate:
    """A crate version that must be locked in a baseline program's Cargo.lock."""

    crate: str
    version: str
    baseline_dir: str
    systems: tuple[str, ...]


LOCKED_CRATES: tuple[LockedCrate, ...] = (
    LockedCrate("sp1-sdk", "6.8.1", "sp1", SP1_SYSTEMS),
    LockedCrate("risc0-zkvm", "3.0.3", "risc0", ("risc0",)),
)


def nargo_path() -> Path:
    return Path(TOOLS[0].command[0])


def bb_path() -> Path:
    return Path(TOOLS[1].command[0])


@dataclass
class Probe:
    name: str
    path: str
    expected: str | None
    found: str | None
    status: str  # "ok" | "recorded" | "mismatch" | "missing" | "not built"
    systems: tuple[str, ...]

    @property
    def failed(self) -> bool:
        return self.status in ("mismatch", "missing")


def _resolve(executable: str) -> str | None:
    if os.sep in executable:
        return executable if Path(executable).is_file() else None
    return shutil.which(executable)


def probe_tool(tool: Tool) -> Probe:
    executable = _resolve(tool.command[0])
    if executable is None:
        return Probe(tool.name, tool.command[0], tool.expected, None, "missing", tool.systems)
    try:
        proc = subprocess.run([executable, *tool.command[1:]], capture_output=True,
                              text=True, timeout=60)
        found = (proc.stdout + proc.stderr).strip().replace("\n", " | ")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Probe(tool.name, executable, tool.expected, f"error: {exc}", "missing",
                     tool.systems)
    if tool.expected is None:
        status = "recorded"
    else:
        status = "ok" if tool.expected in found else "mismatch"
    return Probe(tool.name, executable, tool.expected, found, status, tool.systems)


def _locked_version(lock_file: Path, crate: str) -> str | None:
    text = lock_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf'name = "{re.escape(crate)}"\nversion = "([^"]+)"', text)
    return match.group(1) if match else None


def probe_locked_crate(locked: LockedCrate) -> Probe:
    base = paths.BASELINES / locked.baseline_dir
    locks = sorted(base.glob("**/Cargo.lock")) if base.is_dir() else []
    found = []
    for lock in locks:
        version = _locked_version(lock, locked.crate)
        if version is not None:
            found.append(f"{paths.relative_to_repo(lock)}: {version}")
    name = f"{locked.crate} (Cargo.lock)"
    if not found:
        return Probe(name, str(base), locked.version, None, "not built", locked.systems)
    status = "ok" if all(f.endswith(": " + locked.version) for f in found) else "mismatch"
    return Probe(name, str(base), locked.version, "; ".join(found), status, locked.systems)


def python_packages() -> dict[str, str]:
    """Versions of the Python side (interpreter, NumPy, Z3, maturin)."""
    from importlib import metadata

    info = {"python": sys.version.split()[0], "executable": sys.executable}
    for distribution in ("numpy", "z3-solver", "maturin"):
        try:
            info[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            info[distribution] = "not installed"
    return info


def check_environment() -> list[Probe]:
    """Probe every tool and locked crate."""
    probes = [probe_tool(tool) for tool in TOOLS]
    probes.extend(probe_locked_crate(locked) for locked in LOCKED_CRATES)
    return probes


def failures_for(probes: list[Probe], systems: list[str]) -> list[Probe]:
    """The failed probes that matter for `systems` (tools with no system always matter)."""
    wanted = set(systems)
    return [p for p in probes if p.failed and (not p.systems or wanted & set(p.systems))]


def render(probes: list[Probe]) -> str:
    lines = [f"{'tool':<26} {'status':<10} {'expected':<32} found"]
    for p in probes:
        lines.append(f"{p.name:<26} {p.status:<10} {str(p.expected or '-'):<32} "
                     f"{(p.found or '-')[:110]}")
        lines.append(f"{'':<26} path: {p.path}")
    return "\n".join(lines)


def as_json(probes: list[Probe]) -> list[dict]:
    return [asdict(p) for p in probes]


def main_check_env(systems: list[str], out: Path | None) -> int:
    """`check-env`: print and record every probe; non-zero exit on a failure."""
    probes = check_environment()
    print(render(probes))
    print("python:", json.dumps(python_packages()))
    if out is not None:
        out.write_text(json.dumps({"tools": as_json(probes), "python": python_packages()},
                                  indent=2))
    failed = failures_for(probes, systems)
    if failed:
        print("\nFAILED for the requested systems: "
              + ", ".join(f"{p.name} ({p.status})" for p in failed), file=sys.stderr)
        return 1
    return 0
