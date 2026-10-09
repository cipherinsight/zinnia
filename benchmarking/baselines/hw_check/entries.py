"""The cases a handwritten baseline can be written for, and where its files live.

Two case sets:

* DS-1000 (`datasets/case_sets/ds1000_perf.json`): one folder per case,
  `datasets/ds1000/<case>/`, entry function `kernel`, files `fixture.json`,
  `target.rs`, `main.nr`, `Prover.toml`.
* NPBench/Pythran (`datasets/case_sets/npbench_pythran_perf.json`): one folder
  per program, `datasets/{npbench,pythran}/<program>/`, holding one or more
  entry functions; the case id is `<program>__<function>`. The fixture is the
  case set's (`fixture.<function>.json` for a program with several entry
  functions, else `fixture.json`), and the baseline files carry the function
  name: `target.<function>.rs`, `main.<function>.nr`, `Prover.<function>.toml`.

A case is named on the command line by its id (`ds1000_case0309`,
`pythran_lu__factorMatrix0`) or, for DS-1000, by its number (`0309`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]
DATASETS = BENCH / "datasets"
DS1000 = DATASETS / "ds1000"
DS1000_CASE_SET = DATASETS / "case_sets" / "ds1000_perf.json"
NPB_CASE_SET = DATASETS / "case_sets" / "npbench_pythran_perf.json"
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


def baseline_file_name(name: str, function: str | None) -> str:
    """`target.rs` -> `target.<function>.rs` for an entry of a shape-generic
    program (the harness's `PerfCase.baseline_file` gives the same names)."""
    if function is None:
        return name
    stem, suffix = name.split(".", 1)
    return f"{stem}.{function}.{suffix}"


@dataclass(frozen=True)
class Entry:
    """One case: a DS-1000 case or an NPBench/Pythran entry function."""

    case: str
    #: The case folder (DS-1000) or the program folder (NPBench/Pythran).
    directory: Path
    #: The entry function of a shape-generic program; `None` for DS-1000.
    function: str | None
    fixture: Path
    #: For a function returning `None`: the array arguments it writes, in
    #: parameter order; they are its outputs.
    mutated_outputs: tuple[str, ...] = ()
    atol: float = DEFAULT_ATOL
    rtol: float = DEFAULT_RTOL
    status: str | None = None

    @property
    def kernel(self) -> Path:
        return self.directory / "kernel.py"

    @property
    def kernel_function(self) -> str:
        return self.function or "kernel"

    def file(self, name: str) -> Path:
        """A baseline file of this case: `target.rs`, `main.nr` or `Prover.toml`."""
        return self.directory / baseline_file_name(name, self.function)

    @property
    def target_rs(self) -> Path:
        return self.file("target.rs")

    @property
    def main_nr(self) -> Path:
        return self.file("main.nr")

    @property
    def prover_toml(self) -> Path:
        return self.file("Prover.toml")

    def program_args(self) -> list[str]:
        """`--case <folder> [--function <f>]` for `halo2_hw` and the Noir driver."""
        return ["--case", str(self.directory)] + (["--function", self.function] if self.function else [])


def _tolerance(raw: dict) -> tuple[float, float]:
    tol = raw.get("float_tolerance") or {}
    return tol.get("atol", DEFAULT_ATOL), tol.get("rtol", DEFAULT_RTOL)


@cache
def ds1000_entries() -> dict[str, Entry]:
    """Every DS-1000 case (folder or case-set entry), with its case-set status and tolerance."""
    listed = {}
    if DS1000_CASE_SET.exists():
        listed = {c["case"]: c for c in json.loads(DS1000_CASE_SET.read_text())["cases"]}
    names = sorted({p.name for p in DS1000.iterdir() if p.is_dir()} | set(listed))
    entries = {}
    for name in names:
        raw = listed.get(name, {})
        atol, rtol = _tolerance(raw)
        folder = DS1000 / name
        entries[name] = Entry(name, folder, None, folder / "fixture.json",
                              atol=atol, rtol=rtol, status=raw.get("status"))
    return entries


@cache
def npb_entries() -> dict[str, Entry]:
    """Every entry of the NPBench/Pythran case set (any status)."""
    entries = {}
    for raw in json.loads(NPB_CASE_SET.read_text())["cases"]:
        directory = BENCH / raw["path"]
        atol, rtol = _tolerance(raw)
        entries[raw["case"]] = Entry(raw["case"], directory, raw["function"],
                                     directory / (raw.get("fixture") or "fixture.json"),
                                     tuple(raw.get("mutated_outputs", ())), atol, rtol,
                                     raw.get("status"))
    return entries


def case_name(spec: str) -> str:
    """A case id from a command-line name (a bare number is a DS-1000 case)."""
    return f"ds1000_case{spec}" if spec.isdigit() else spec


def resolve(spec: str) -> Entry:
    name = case_name(spec)
    for entries in (npb_entries(), ds1000_entries()):
        if name in entries:
            return entries[name]
    raise KeyError(f"unknown case {spec!r} (neither a DS-1000 case nor an NPBench/Pythran entry)")


def with_baselines(entries: dict[str, Entry]) -> list[Entry]:
    """The entries with a `target.rs` or a `main.nr`, in case order."""
    return [e for _, e in sorted(entries.items()) if e.target_rs.exists() or e.main_nr.exists()]
