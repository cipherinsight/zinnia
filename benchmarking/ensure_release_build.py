"""Helpers for making benchmark runs use an optimized Zinnia extension.

The Python runners call into the installed PyO3 module. If a developer last
ran `maturin develop` without `--release`, proof timings can silently measure
debug Rust code. Benchmarks should always install the release extension first.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
ROOT = THIS_DIR.parent
READY_ENV = "ZINNIA_RELEASE_BUILD_READY"
SKIP_ENV = "ZINNIA_SKIP_RELEASE_BUILD"
FEATURES_ENV = "ZINNIA_RELEASE_FEATURES"
DEFAULT_FEATURES = "halo2-base-backend"
DEFAULT_TOOLCHAIN = "nightly-2025-12-01"


def _maturin_bin() -> str:
    candidate = ROOT / ".venv" / "bin" / "maturin"
    if candidate.exists():
        return str(candidate)
    return "maturin"


def ensure_zinnia_release_build() -> None:
    """Install the local Zinnia extension from a release Rust build.

    Set `ZINNIA_SKIP_RELEASE_BUILD=1` only for intentionally prebuilt
    environments. Parent runners set `ZINNIA_RELEASE_BUILD_READY=1` after a
    successful install so subprocess-per-case harnesses do not rebuild for
    every case.
    """
    if os.environ.get(SKIP_ENV) == "1" or os.environ.get(READY_ENV) == "1":
        return

    features = os.environ.get(FEATURES_ENV, DEFAULT_FEATURES).strip()
    cmd = [_maturin_bin(), "develop", "--release"]
    if features:
        cmd.extend(["--features", features])

    env = os.environ.copy()
    env.setdefault("RUSTUP_TOOLCHAIN", DEFAULT_TOOLCHAIN)
    print(
        "Ensuring Zinnia release Rust build: "
        + " ".join(cmd)
        + f" (RUSTUP_TOOLCHAIN={env['RUSTUP_TOOLCHAIN']})",
        flush=True,
    )
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise RuntimeError(
            f"release build failed with rc={proc.returncode}; "
            f"command={' '.join(cmd)}"
        )
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    os.environ[READY_ENV] = "1"
