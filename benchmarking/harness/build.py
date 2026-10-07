"""The release build of the Zinnia extension and its build record (audit H-01).

`build` compiles the extension with both benchmark backends in the release
profile, the way `scripts/develop.sh` links the libz3 shipped in the venv's
z3-solver wheel, and writes a build record (`target/zinnia-harness-build.json`)
with the `.so` SHA-256, profile, features, rustc and source state.

`run` refuses to time anything unless the installed `.so` is the one the record
describes, is byte-identical to Cargo's release artifact, and is not the debug
artifact.

`build_noir_execute` builds Noir's artifact executor from the Noir commit the
pinned nargo reports and records it (`target/zinnia-harness-noir-execute.json`);
`run` refuses a Noir system unless the installed binary is that build
(`toolchains.probe_noir_execute_build`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from benchmarking.harness import gitstate, paths, toolchains

FEATURES = "halo2-base-backend,noir-backend"
PROFILE = "release"
DEFAULT_BUILD_JOBS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_rustc_version(path: Path) -> str | None:
    """The `rustc version ...` string the compiler embeds in the binary."""
    data = path.read_bytes()
    match = re.search(rb"rustc version [0-9][\x20-\x7e]*", data)
    return match.group(0).decode() if match else None


def has_debug_sections(path: Path) -> bool | None:
    try:
        proc = subprocess.run(["readelf", "-S", str(path)], capture_output=True, text=True,
                              timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ".debug_info" in proc.stdout


def _command_version(*args: str) -> str | None:
    try:
        proc = subprocess.run(list(args), capture_output=True, text=True, timeout=60,
                              cwd=paths.REPO_ROOT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def build_release(jobs: int = DEFAULT_BUILD_JOBS) -> dict:
    """Build and install the release extension; write and return the build record."""
    env = dict(os.environ)
    env.update({
        "RUSTC_BOOTSTRAP": "1",  # the poseidon-primitives dependency uses #![feature]
        "VIRTUAL_ENV": str(paths.REPO_ROOT / ".venv"),
        "PYTHON": sys.executable,
        "CARGO_BUILD_JOBS": str(jobs),
    })
    command = ["nice", "-n", "10", str(paths.REPO_ROOT / "scripts" / "develop.sh"),
               "--release", "--features", FEATURES]
    print("building:", " ".join(command), flush=True)
    started = time.time()
    proc = subprocess.run(command, cwd=paths.REPO_ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"release build failed (exit {proc.returncode})")
    record = describe_build(command, time.time() - started)
    paths.BUILD_RECORD.write_text(json.dumps(record, indent=2))
    print(f"build record: {paths.BUILD_RECORD}")
    print(json.dumps(record, indent=2))
    return record


def describe_build(command: list[str], seconds: float) -> dict:
    so = paths.installed_extension()
    if so is None:
        raise RuntimeError("the build finished but no extension module is installed")
    return {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(seconds, 1),
        "command": command,
        "profile": PROFILE,
        "features": FEATURES,
        "rustc": _command_version("rustc", "--version"),
        "cargo": _command_version("cargo", "--version"),
        "extension": paths.relative_to_repo(so),
        "extension_sha256": sha256_file(so),
        "embedded_rustc": embedded_rustc_version(so),
        "source": gitstate.describe(paths.REPO_ROOT),
    }


def installed_build_problems() -> tuple[dict | None, list[str]]:
    """The build record and every reason the installed `.so` may not be timed."""
    problems: list[str] = []
    so = paths.installed_extension()
    if so is None:
        return None, ["no extension module installed (run `python -m benchmarking.harness build`)"]
    if not paths.BUILD_RECORD.exists():
        return None, [f"no build record at {paths.BUILD_RECORD} "
                      "(run `python -m benchmarking.harness build`)"]
    record = json.loads(paths.BUILD_RECORD.read_text())
    installed = sha256_file(so)
    if record.get("extension_sha256") != installed:
        problems.append("installed .so differs from the one the build record describes")
    if record.get("profile") != PROFILE or record.get("features") != FEATURES:
        problems.append(f"build record says profile={record.get('profile')} "
                        f"features={record.get('features')}")
    if not paths.RELEASE_ARTIFACT.exists() or sha256_file(paths.RELEASE_ARTIFACT) != installed:
        problems.append("installed .so is not Cargo's release artifact")
    if paths.DEBUG_ARTIFACT.exists() and sha256_file(paths.DEBUG_ARTIFACT) == installed:
        problems.append("installed .so is the debug artifact")
    if has_debug_sections(so):
        problems.append("installed .so carries debug sections")
    return record, problems


# ---------------------------------------------------------------------------
# Baseline programs
# ---------------------------------------------------------------------------

#: How each baseline program directory is built (its README's command), release
#: profile. `halo2_hw` builds the cases `HALO2_HW_CASES` names, or every case
#: with a `target.rs`; the variable is recorded.
BASELINE_BUILDS: dict[str, list[str]] = {
    "halo2_hw": ["cargo", "build", "--release"],
    "sp1": ["cargo", "build", "--release"],
    "risc0": ["cargo", "build", "--release"],
    "cairo": [str(toolchains.tool_executable("scarb")), "build"],
}
BASELINE_RECORD = paths.REPO_ROOT / "target" / "zinnia-harness-baselines.json"


def build_baselines(names: list[str], jobs: int = DEFAULT_BUILD_JOBS) -> dict:
    """Build baseline programs one at a time; record command, toolchain and outputs."""
    record = json.loads(BASELINE_RECORD.read_text()) if BASELINE_RECORD.exists() else {}
    for name in names:
        directory = paths.BASELINES / name
        env = dict(os.environ, CARGO_BUILD_JOBS=str(jobs))
        env["PATH"] = f"{Path.home() / '.sp1' / 'bin'}:{env.get('PATH', '')}"
        command = ["nice", "-n", "10", *BASELINE_BUILDS[name]]
        print(f"building {name}: {' '.join(command)} (in {directory})", flush=True)
        started = time.time()
        proc = subprocess.run(command, cwd=directory, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"building {name} failed (exit {proc.returncode})")
        binaries = sorted((directory / "target" / "release").glob("*")) \
            if (directory / "target" / "release").is_dir() else []
        record[name] = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": round(time.time() - started, 1),
            "command": command,
            "HALO2_HW_CASES": os.environ.get("HALO2_HW_CASES"),
            "rustc": _command_version("rustc", "--version"),
            "executables_sha256": {paths.relative_to_repo(b): sha256_file(b) for b in binaries
                                   if b.is_file() and os.access(b, os.X_OK)},
        }
    BASELINE_RECORD.write_text(json.dumps(record, indent=2))
    print(f"baseline build record: {BASELINE_RECORD}")
    return record


# ---------------------------------------------------------------------------
# Noir's artifact executor
# ---------------------------------------------------------------------------

NOIR_REPOSITORY = "https://github.com/noir-lang/noir.git"
NOIR_EXECUTE_PACKAGE = "noir_artifact_cli"
NOIR_EXECUTE_BIN = "noir-execute"


def _run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("  $", " ".join(command), flush=True)
    proc = subprocess.run(command, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed (exit {proc.returncode})")


def _git_output(source: Path, *args: str) -> str | None:
    proc = subprocess.run(["git", *args], cwd=source, capture_output=True, text=True,
                          timeout=120)
    return proc.stdout.strip() if proc.returncode == 0 else None


def checkout_noir(commit: str, source: Path = paths.NOIR_EXECUTE_SOURCE) -> None:
    """A clean checkout of `commit` of the Noir repository at `source`.

    Fetches only that commit (`--depth 1`). An existing checkout is reused
    when it is clean; a modified one is refused rather than overwritten.
    """
    if not (source / ".git").exists():
        source.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q"], source)
        _run(["git", "remote", "add", "origin", NOIR_REPOSITORY], source)
    if _git_output(source, "status", "--porcelain"):
        raise RuntimeError(f"{source} has local changes; delete it to start from a fresh clone")
    if _git_output(source, "rev-parse", "HEAD") != commit:
        _run(["git", "fetch", "-q", "--depth", "1", "origin", commit], source)
        _run(["git", "checkout", "-q", "--detach", commit], source)
    head = _git_output(source, "rev-parse", "HEAD")
    if head != commit:
        raise RuntimeError(f"{source} is at {head} after checking out {commit}")


def rust_toolchain_channel(source: Path) -> str:
    """The Rust toolchain Noir pins in its `rust-toolchain.toml`."""
    config = tomllib.loads((source / "rust-toolchain.toml").read_text())
    return config["toolchain"]["channel"]


def build_noir_execute(jobs: int = DEFAULT_BUILD_JOBS) -> dict:
    """Build `noir-execute` from the Noir commit of the pinned nargo; install and record it.

    The build uses the Rust toolchain Noir's checkout pins (installed with
    rustup's minimal profile, without the extra targets the toolchain file
    lists) and Noir's `Cargo.lock` (`--locked`).
    """
    commit = toolchains.nargo_commit()
    source = paths.NOIR_EXECUTE_SOURCE
    print(f"noir-execute: Noir {commit} in {source}", flush=True)
    started = time.time()
    checkout_noir(commit, source)
    channel = rust_toolchain_channel(source)
    if shutil.which("rustup") is None:
        raise RuntimeError(f"rustup is needed to install Noir's Rust toolchain {channel}")
    _run(["rustup", "toolchain", "install", channel, "--profile", "minimal",
          "--no-self-update"], source)
    command = ["nice", "-n", "10", "cargo", f"+{channel}", "build", "--release", "--locked",
               "-p", NOIR_EXECUTE_PACKAGE, "--bin", NOIR_EXECUTE_BIN]
    _run(command, source, env=dict(os.environ, CARGO_BUILD_JOBS=str(jobs)))

    binary = paths.NOIR_EXECUTE_BINARY
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "target" / "release" / NOIR_EXECUTE_BIN, binary)
    version = _command_version(str(binary), "--version")
    if toolchains.NOIR_VERSION not in (version or ""):
        raise RuntimeError(f"{binary} --version says {version!r}, not {toolchains.NOIR_VERSION}")
    record = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(time.time() - started, 1),
        "noir_repository": NOIR_REPOSITORY,
        "noir_commit": commit,
        "version": version,
        "nargo_version": _command_version(str(toolchains.nargo_path()), "--version"),
        "command": command,
        "rust_toolchain": channel,
        "rustc": _command_version("rustc", f"+{channel}", "--version"),
        "cargo": _command_version("cargo", f"+{channel}", "--version"),
        "RUSTFLAGS": os.environ.get("RUSTFLAGS"),
        "binary": paths.relative_to_repo(binary),
        "binary_sha256": sha256_file(binary),
        "source": gitstate.describe(source),
    }
    paths.NOIR_EXECUTE_RECORD.write_text(json.dumps(record, indent=2))
    print(f"noir-execute build record: {paths.NOIR_EXECUTE_RECORD}")
    print(json.dumps(record, indent=2))
    probe = toolchains.probe_noir_execute_build()
    if probe.status != "ok":
        raise RuntimeError(f"the built noir-execute does not pass its check: {probe.found}")
    return record
