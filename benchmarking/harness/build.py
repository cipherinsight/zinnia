"""The release build of the Zinnia extension and its build record (audit H-01).

`build` compiles the extension with both benchmark backends in the release
profile, the way `scripts/develop.sh` links the libz3 shipped in the venv's
z3-solver wheel, and writes a build record (`target/zinnia-harness-build.json`)
with the `.so` SHA-256, profile, features, rustc and source state.

`run` refuses to time anything unless the installed `.so` is the one the record
describes, is byte-identical to Cargo's release artifact, and is not the debug
artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
