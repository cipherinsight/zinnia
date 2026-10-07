"""Where things live. Every other module takes its paths from here."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKING = REPO_ROOT / "benchmarking"

DATASETS = BENCHMARKING / "datasets"
DS1000_PERF = DATASETS / "ds1000"
CASE_SETS = DATASETS / "case_sets"
PERF_CASE_SET = CASE_SETS / "ds1000_perf.json"
ACCEPTANCE_CASE_SET = CASE_SETS / "acceptance.json"

#: The baseline programs. `ZINNIA_HARNESS_BASELINES` points elsewhere (the
#: harness tests use it for fake programs); the manifest records the variable.
BASELINES = Path(os.environ.get("ZINNIA_HARNESS_BASELINES", BENCHMARKING / "baselines"))
#: The Q32 fixed-point Noir library shared by both Noir sides. Zinnia's Noir
#: backend embeds the same files with `include_str!` (src/prove/noir/mod.rs).
NOIR_FPC = BENCHMARKING / "noir_fpc"

RESULTS = BENCHMARKING / "results"

#: The installed extension module (built by `python -m benchmarking.harness build`).
EXTENSION_DIR = REPO_ROOT / "zinnia"
EXTENSION_GLOB = "_zinnia_core*.so"
#: Cargo's release and debug artifacts of the extension.
RELEASE_ARTIFACT = REPO_ROOT / "target" / "release" / "lib_zinnia_core.so"
DEBUG_ARTIFACT = REPO_ROOT / "target" / "debug" / "lib_zinnia_core.so"
#: Written by the `build` command; checked by `run` before any timing.
BUILD_RECORD = REPO_ROOT / "target" / "zinnia-harness-build.json"

#: Noir's artifact executor (`noir-execute`), built by `build --noir-execute`
#: from the Noir commit the pinned nargo reports: the Noir checkout, the
#: installed binary and its build record.
NOIR_EXECUTE_DIR = REPO_ROOT / "target" / "noir-execute"
NOIR_EXECUTE_SOURCE = NOIR_EXECUTE_DIR / "noir"
NOIR_EXECUTE_BINARY = NOIR_EXECUTE_DIR / "bin" / "noir-execute"
NOIR_EXECUTE_RECORD = REPO_ROOT / "target" / "zinnia-harness-noir-execute.json"


def installed_extension() -> Path | None:
    """The extension module Python imports, or `None` if there is none."""
    matches = sorted(EXTENSION_DIR.glob(EXTENSION_GLOB))
    return matches[0] if matches else None


def relative_to_repo(path: Path) -> str:
    """`path` relative to the repository root when it is inside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
