"""Case-set files under `datasets/case_sets/` (written by the dataset workstream).

Expected shape of the performance set (`ds1000_perf.json`):

```json
{"cases": [{"case": "ds1000_case0309", "status": "included",
            "reason": null, "tolerance": {"abs": 1e-6, "rel": 0.0}}, ...]}
```

`status` other than `included` excludes the case (its `reason` is recorded).
`tolerance` applies to float outputs: |actual - expected| <= abs + rel*|expected|.
A case without a tolerance uses `DEFAULT_TOLERANCE`, and the record says so.

Expected shape of the acceptance set (`acceptance.json`):

```json
{"programs": [{"program": "npbench_azimint_hist", "dataset": "npbench",
               "kernel": "datasets/npbench/npbench_azimint_hist/kernel.py",
               "function": "kernel", "fixture": "datasets/.../fixture.json" | null,
               "tolerance": {...} | null}, ...]}
```

Paths are relative to `benchmarking/`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarking.harness import paths

INCLUDED = "included"


@dataclass(frozen=True)
class Tolerance:
    abs: float
    rel: float
    source: str

    def allows(self, expected: float, actual: float) -> bool:
        return abs(actual - expected) <= self.abs + self.rel * abs(expected)


#: Used when the case set gives no tolerance; recorded as such in every result.
DEFAULT_TOLERANCE = Tolerance(abs=1e-6, rel=1e-6, source="harness default")


def parse_tolerance(raw: Any, source: str) -> Tolerance:
    if raw is None:
        return DEFAULT_TOLERANCE
    if isinstance(raw, (int, float)):
        return Tolerance(abs=float(raw), rel=0.0, source=source)
    absolute = raw.get("abs", raw.get("atol", 0.0))
    relative = raw.get("rel", raw.get("rtol", 0.0))
    return Tolerance(abs=float(absolute), rel=float(relative), source=source)


@dataclass(frozen=True)
class PerfCase:
    case: str
    directory: Path
    fixture_path: Path
    function: str
    tolerance: Tolerance
    included: bool
    reason: str | None

    @property
    def kernel_path(self) -> Path:
        return self.directory / "kernel.py"

    def to_json(self) -> dict:
        return {"case": self.case, "directory": str(self.directory),
                "fixture": str(self.fixture_path), "function": self.function,
                "tolerance": vars(self.tolerance), "included": self.included,
                "reason": self.reason}

    @staticmethod
    def from_json(raw: dict) -> "PerfCase":
        return PerfCase(case=raw["case"], directory=Path(raw["directory"]),
                        fixture_path=Path(raw["fixture"]), function=raw["function"],
                        tolerance=Tolerance(**raw["tolerance"]), included=raw["included"],
                        reason=raw["reason"])


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perf_case(entry: Any, fixtures_dir: Path | None) -> PerfCase:
    if isinstance(entry, str):
        entry = {"case": entry}
    case = entry["case"]
    directory = paths.BENCHMARKING / entry["path"] if "path" in entry else paths.DS1000_PERF / case
    if fixtures_dir is not None:
        fixture = fixtures_dir / case / "fixture.json"
    elif "fixture" in entry:
        fixture = paths.BENCHMARKING / entry["fixture"]
    else:
        fixture = directory / "fixture.json"
    status = entry.get("status", INCLUDED)
    return PerfCase(case=case, directory=directory, fixture_path=fixture,
                    function=entry.get("function", "kernel"),
                    tolerance=parse_tolerance(entry.get("tolerance"), "case set"),
                    included=status == INCLUDED,
                    reason=None if status == INCLUDED else entry.get("reason", status))


def load_perf_cases(case_set: Path | None, case_ids: list[str] | None,
                    fixtures_dir: Path | None) -> list[PerfCase]:
    """The cases of a run.

    `case_ids` alone (no case set) runs those DS-1000 folders with the default
    tolerance; with a case set it selects among the set's cases. `fixtures_dir`
    reads `<fixtures_dir>/<case>/fixture.json` instead of the case folder's
    fixture (for trial runs before the dataset workstream commits fixtures).
    """
    if case_set is not None:
        raw = json.loads(case_set.read_text())
        entries = raw["cases"] if isinstance(raw, dict) else raw
        cases = [_perf_case(entry, fixtures_dir) for entry in entries]
        if case_ids:
            wanted = set(case_ids)
            cases = [c for c in cases if c.case in wanted]
            unknown = wanted - {c.case for c in cases}
            if unknown:
                raise ValueError(f"cases not in {case_set}: {sorted(unknown)}")
        return cases
    if not case_ids:
        raise ValueError("give a case set or explicit cases")
    return [_perf_case(case, fixtures_dir) for case in case_ids]


@dataclass(frozen=True)
class AcceptanceProgram:
    program: str
    dataset: str
    kernel_path: Path
    function: str
    fixture_path: Path | None
    tolerance: Tolerance


def load_acceptance_programs(case_set: Path) -> list[AcceptanceProgram]:
    raw = json.loads(case_set.read_text())
    entries = raw["programs"] if isinstance(raw, dict) else raw
    programs = []
    for entry in entries:
        kernel = entry.get("kernel") or entry.get("kernel_path")
        fixture = entry.get("fixture")
        programs.append(AcceptanceProgram(
            program=entry["program"], dataset=entry.get("dataset", entry["program"].split("_")[0]),
            kernel_path=paths.BENCHMARKING / kernel, function=entry.get("function", "kernel"),
            fixture_path=paths.BENCHMARKING / fixture if fixture else None,
            tolerance=parse_tolerance(entry.get("tolerance"), "case set")))
    return programs
