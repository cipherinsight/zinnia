"""Case-set files under `datasets/case_sets/` (written by the dataset workstream).

Shape of the performance set (`ds1000_perf.json`, written by `datasets/tools`):

```json
{"cases": [{"case": "ds1000_case0309", "status": "perf", "reason": "...",
            "fixture": "fixture.json", "data_movement_only": false,
            "float_tolerance": {"atol": 1e-6, "rtol": 1e-6}}, ...]}
```

Only `status == "perf"` cases run; any other status (`duplicate`, `degenerate`,
`out_of_scope`, `zinnia_rejects`, ...) excludes the case and its `reason` is
recorded. `fixture` is relative to the case folder. `float_tolerance` applies to
float outputs: |actual - expected| <= atol + rtol*|expected|; a case without one
uses `DEFAULT_TOLERANCE`, and the record says so. `data_movement_only` cases only
copy inputs; the analysis reports them apart from the headline.

Optional fields, used by the NPBench/Pythran set (`npbench_pythran_perf.json`,
written by `datasets/tools/npbench_pythran_perf.py`):

- `path`: the case folder relative to `benchmarking/` (default
  `datasets/ds1000/<case>`); `function`: the kernel function (default `kernel`).
- `shape_generic: true`: the folder holds a program with named entry functions
  (one case per function); the zkVM hosts and `halo2_hw` get `--function
  <function>`, and the handwritten circuits are per function
  (`PerfCase.baseline_file`: `target.<function>.rs`, `main.<function>.nr`).
- `mutated_outputs: [names]`: for a function that returns `None`, the array
  arguments it writes, in parameter order: they are its outputs.
- `group`: a label (e.g. the corpus); the analysis also compares per group.

Shape of the acceptance set (`acceptance.json`):

```json
{"entries": [{"program": "pythran_cronbach", "dataset": "pythran",
              "function": "cronbach", "entry_function": true, "status": "in_scope",
              "kernel": "pythran/pythran_cronbach/kernel.py",
              "fixture": "pythran/pythran_cronbach/fixture.json",
              "float_tolerance": {"atol": 1e-6, "rtol": 1e-6}}, ...]}
```

Only entry functions with `status == "in_scope"` are runnable programs; helper
functions and out-of-scope programs are not. Kernel and fixture paths are
relative to `benchmarking/datasets/`.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarking.harness import paths

#: The perf-set status of a case that runs; every other status excludes it.
INCLUDED = "perf"


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
    data_movement_only: bool = False
    #: The case folder holds a program with several named entry functions.
    shape_generic: bool = False
    #: A function returning `None` outputs these written array arguments.
    mutated_outputs: tuple[str, ...] = ()
    group: str | None = None

    @property
    def kernel_path(self) -> Path:
        return self.directory / "kernel.py"

    def baseline_file(self, name: str, variant: str = "") -> Path:
        """A handwritten circuit of this case: `target.rs` / `main.nr` in a
        DS-1000 folder, `target.<function>.rs` / `main.<function>.nr` in a
        shape-generic program folder (the names `baselines/hw_check/entries.py`
        uses); a variant such as `literal` adds its tag before the suffix
        (`target.literal.rs`, `main.<function>.literal.nr`)."""
        stem, suffix = name.split(".", 1)
        tags = [t for t in (self.function if self.shape_generic else None, variant) if t]
        return self.directory / ".".join([stem, *tags, suffix])

    def to_json(self) -> dict:
        return {"case": self.case, "directory": str(self.directory),
                "fixture": str(self.fixture_path), "function": self.function,
                "tolerance": vars(self.tolerance), "included": self.included,
                "reason": self.reason, "data_movement_only": self.data_movement_only,
                "shape_generic": self.shape_generic,
                "mutated_outputs": list(self.mutated_outputs), "group": self.group}

    @staticmethod
    def from_json(raw: dict) -> "PerfCase":
        return PerfCase(case=raw["case"], directory=Path(raw["directory"]),
                        fixture_path=Path(raw["fixture"]), function=raw["function"],
                        tolerance=Tolerance(**raw["tolerance"]), included=raw["included"],
                        reason=raw["reason"],
                        data_movement_only=raw.get("data_movement_only", False),
                        shape_generic=raw.get("shape_generic", False),
                        mutated_outputs=tuple(raw.get("mutated_outputs", ())),
                        group=raw.get("group"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perf_case(entry: Any, fixtures_dir: Path | None) -> PerfCase:
    if isinstance(entry, str):
        entry = {"case": entry}
    case = entry["case"]
    directory = paths.BENCHMARKING / entry["path"] if "path" in entry else paths.DS1000_PERF / case
    if fixtures_dir is not None:
        fixture = fixtures_dir / case / "fixture.json"
    else:
        fixture = directory / (entry.get("fixture") or "fixture.json")
    status = entry.get("status", INCLUDED)
    tolerance = entry.get("float_tolerance", entry.get("tolerance"))
    return PerfCase(case=case, directory=directory, fixture_path=fixture,
                    function=entry.get("function", "kernel"),
                    tolerance=parse_tolerance(tolerance, "case set"),
                    included=status == INCLUDED,
                    reason=None if status == INCLUDED else entry.get("reason", status),
                    data_movement_only=bool(entry.get("data_movement_only", False)),
                    shape_generic=bool(entry.get("shape_generic", False)),
                    mutated_outputs=tuple(entry.get("mutated_outputs", ())),
                    group=entry.get("group"))


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


def parse_shard(text: str) -> tuple[int, int]:
    """`K/N` -> (K, N): shard K (counting from 0) of N."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise ValueError(f"shard {text!r} is not K/N") from None
    if not 0 <= index < count:
        raise ValueError(f"shard {text!r}: need 0 <= K < N")
    return index, count


def shard_cases(cases: list[PerfCase], index: int, count: int) -> list[PerfCase]:
    """The cases of shard `index` of `count`.

    The included cases, sorted by id, are dealt out in turn: case number `i`
    goes to shard `i % count`. The assignment depends only on the case ids, so
    the shards of one case set are disjoint and cover it whatever the seed.
    Excluded cases stay in every shard (they are records, not runs).
    """
    included = sorted((c for c in cases if c.included), key=lambda c: c.case)
    mine = {c.case for position, c in enumerate(included) if position % count == index}
    return [c for c in cases if not c.included or c.case in mine]


@dataclass(frozen=True)
class AcceptanceProgram:
    program: str
    dataset: str
    kernel_path: Path
    function: str
    fixture_path: Path | None
    tolerance: Tolerance


def load_acceptance_programs(case_set: Path) -> list[AcceptanceProgram]:
    """The runnable programs of an acceptance set: in-scope entry functions."""
    raw = json.loads(case_set.read_text())
    entries = raw["entries"] if isinstance(raw, dict) else raw
    programs = []
    for entry in entries:
        if entry.get("status", "in_scope") != "in_scope" or not entry.get("entry_function", True):
            continue
        fixture = entry.get("fixture")
        tolerance = entry.get("float_tolerance", entry.get("tolerance"))
        programs.append(AcceptanceProgram(
            program=entry["program"], dataset=entry.get("dataset", entry["program"].split("_")[0]),
            kernel_path=paths.DATASETS / entry["kernel"], function=entry.get("function", "kernel"),
            fixture_path=paths.DATASETS / fixture if fixture else None,
            tolerance=parse_tolerance(tolerance, "case set")))
    return programs
