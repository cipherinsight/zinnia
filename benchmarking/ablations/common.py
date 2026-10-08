"""Shared parts of the ablation scripts: the configurations, one measured compile in a
child process, and the summary of the layer report (`ZINNIA_LAYER_REPORT`).

A measurement compiles one perf case for one target in its own process, with the
configuration's environment variables set (they are read by the compiler itself,
`docs/ablations.md`), and records:

* `ir`: SHA-256 (16 hex digits) of the exported IR statements, as
  `benchmarking/determinism.py` computes it;
* halo2: `cells` = advice + lookup cells at the `k` that `estimate_params` picks,
  with `k`, `lookup_bits` and `poly_cost` (committed polynomials x 2^k, the size model
  `estimate_params` minimises; it also counts the columns that fixed overheads such as
  a verifier challenge or a lookup table add, which cells do not see)
  (`halo2_base_cell_stats`, witness-free);
* noir: `noir` = SHA-256 (16 hex digits) of the emitted `main.nr` (input B), and with
  gates on, `gates` = `bb gates` circuit size (`noir_gates.gate_count`);
* `layer`: what the array-lowering layer chose (`summarize_report`).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKING = ROOT / "benchmarking"
CASE_SETS = {
    "ds1000": BENCHMARKING / "datasets" / "case_sets" / "ds1000_perf.json",
    "npbench_pythran": BENCHMARKING / "datasets" / "case_sets" / "npbench_pythran_perf.json",
}
TARGETS = ("halo2", "noir")

#: The compiler configurations of the ablation study: environment variables set for the
#: compile (`docs/ablations.md`). `final` is the compiler as measured everywhere else.
CONFIGS: dict[str, dict[str, str]] = {
    "final": {},
    "abl_joint": {"ZINNIA_ABLATE_JOINT": "1"},
    "fix_delegate": {"ZINNIA_FORCE_CONSTRUCTION": "@delegate"},
    "fix_selects": {"ZINNIA_FORCE_CONSTRUCTION": "@selects"},
    "abl_array_facts": {"ZINNIA_ABLATE_ARRAY_FACTS": "1"},
}

#: Constructions through the backend's memory: Noir's native ROM / RAM, halo2-base's
#: dynamic-lookup table read and RAM trace.
NATIVE_MEMORY = ("memory: element by table read", "memory: native RAM", "memory: RAM trace")

#: Constructions that are part of another one, not an alternative to it
#: (`src/lower/forcing.rs`, HELPERS).
HELPERS = ("memory: index bits",)


def perf_entries(case_set: Path) -> dict[str, dict]:
    """The case set's `perf` entries by case name."""
    entries = json.loads(case_set.read_text())["cases"]
    return {e["case"]: e for e in entries if e["status"] == "perf"}


def case_files(entry: dict) -> tuple[Path, Path, str]:
    """`(kernel.py, fixture.json, entry function)` of a case-set entry."""
    directory = (BENCHMARKING / entry["path"] if "path" in entry
                 else BENCHMARKING / "datasets" / "ds1000" / entry["case"])
    fixture = directory / (entry.get("fixture") or "fixture.json")
    return directory / "kernel.py", fixture, entry.get("function", "kernel")


def sha16(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


# ----------------------------------------------------------------- the child side


def measure(kernel: str, fixture_path: str, function: str, target: str, gates: bool) -> dict:
    """Compile once in this process and measure (the configuration is in os.environ)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BENCHMARKING))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    started = time.perf_counter()
    compiled = compile_kernel(Path(kernel), function, backend=target)
    record: dict = {"compile_s": round(time.perf_counter() - started, 3)}
    stmts = compiled.get_ir_stmts()
    record.update(ir=sha16(json.dumps(stmts, sort_keys=True)), stmts=len(stmts))
    if target == "halo2":
        from zinnia._zinnia_core import halo2_base_cell_stats
        stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
        record.update(cells=stats["advice"] + stats["lookup"], advice=stats["advice"],
                      lookup=stats["lookup"], k=stats["k"], lookup_bits=stats["lookup_bits"],
                      poly_cost=stats.get("poly_cost"))
    else:
        fixture = load_fixture(Path(fixture_path))
        source = compiled.emit_structural(*fixture.arguments("B"), backend="noir")
        record["noir"] = sha16(source)
        if gates:
            from noir_gates import NOIR_FPC, gate_count
            record.update(gate_count(source, "noir_fpc", NOIR_FPC))
    return record


def child_main(argv: list[str]) -> int:
    """`--child KERNEL FIXTURE FUNCTION TARGET [--gates]`: print one JSON line."""
    kernel, fixture, function, target = argv[:4]
    try:
        record = measure(kernel, fixture, function, target, "--gates" in argv[4:])
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, KeyboardInterrupt):
            raise
        record = {"error": f"{type(exc).__name__}: {exc}"[:1500]}
    print(json.dumps(record), flush=True)
    return 0


# ---------------------------------------------------------------- the parent side


def run_child(entry: dict, target: str, env: dict[str, str], gates: bool, timeout: float,
              wrap: list[str]) -> dict:
    """Measure one case and target in a child process with `env` added to the
    environment; the layer report is collected and summarized.

    `wrap` (a memory cap such as `capped.py MEM SECONDS`) should enforce its own
    timeout below `timeout`: on this process's timeout only the wrapper is killed, and
    a wrapper that runs its command in a session of its own leaves it running."""
    kernel, fixture, function = case_files(entry)
    with tempfile.TemporaryDirectory(prefix="zinnia-ablation-") as tmp:
        report = Path(tmp) / "layer.jsonl"
        child_env = {**os.environ, **env, "ZINNIA_LAYER_REPORT": str(report),
                     "ZINNIA_SKIP_RELEASE_BUILD": "1"}
        cmd = [*wrap, sys.executable, str(Path(__file__).resolve()), "--child", str(kernel),
               str(fixture), function, target, *(["--gates"] if gates else [])]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT,
                                  env=child_env)
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout:.0f} s"}
        record = None
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.startswith("{"):
                record = json.loads(line)
                break
        if record is None:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            record = {"error": f"exit {proc.returncode}: {' | '.join(tail)}"[:1500]}
        if report.exists():
            record["layer"] = summarize_report(report.read_text().splitlines())
    return record


def summarize_report(lines: list[str]) -> dict:
    """What the layer chose, over every lowering round of one compile:

    * `groups`: per group (in order) its family, the constructions applicable to it,
      the forced one, whether it was released, whether a configuration fixes its family,
      and the constructions chosen for its lanes;
    * `chosen`: per family, how many groups chose each construction;
    * `native_memory`: groups whose lanes come from the backend's memory (NATIVE_MEMORY);
    * `fallback`: groups of a family a named configuration fixes whose lanes come from
      none of the configuration's constructions (the layer's choice is kept for them);
    * `estimated_cost`: the layer's own cost estimate of what it selected, summed over
      rounds (halo2-base cells or Noir gates of the cost model, not a measurement).
    """
    groups: list[dict] = []
    chosen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    estimated = 0
    native = fallback = 0
    for line in lines:
        if not line.strip():
            continue
        round_ = json.loads(line)
        estimated += round_["estimated_cost"]
        forcing = set(round_.get("forcing", []))
        for g in round_["groups"]:
            groups.append({k: g[k] for k in ("family", "kind", "lanes", "applicable", "forced",
                                             "released", "fixed_family", "chosen")})
            for name in g["chosen"]:
                chosen[g["family"]][name] += 1
            if any(name in NATIVE_MEMORY for name in g["chosen"]):
                native += 1
            if g["fixed_family"] and not forcing & set(g["chosen"]):
                fallback += 1
    return {"groups": groups, "chosen": {f: dict(c) for f, c in sorted(chosen.items())},
            "native_memory": native, "fallback": fallback, "estimated_cost": estimated}


def cost_of(record: dict, target: str, metric: str = "cells") -> int | None:
    """The measured cost: halo2-base cells (or `metric`, e.g. `poly_cost`), or Noir gates."""
    if "error" in record:
        return None
    return record.get(metric) if target == "halo2" else record.get("gates")


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    sys.exit(child_main(sys.argv[2:]))
