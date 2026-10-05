"""Deterministic halo2-base cost table: Zinnia against the handwritten halo2_hw baselines.

For every DS-1000 case that has a handwritten halo2-base baseline (`target.rs`):

* **Zinnia**: compile `kernel.py`, let the halo2-base backend's `estimate_params`
  choose `(k, lookup_bits)`, and count the circuit's advice, lookup-advice and
  fixed cells at that lookup width (a witness-free synthesis; cell counts do not
  depend on the inputs).
* **halo2_hw**: `halo2_hw stats` with input `A` of the case's fixture (the
  baseline's own copy of the same sizing rule).

The cost metric is advice + lookup cells (the cells the prover commits to) and
the chosen `k`. The ratio is Zinnia / halo2_hw; the summary gives its geometric
mean over the `perf` cases and over every case measured (functional duplicates
included). Proving time is measured by the harness, not here.

Usage:

    python -m benchmarking.harness.costs_halo2 [--cases 0309,0310] [--out costs.json]
        [--halo2-bin PATH] [--baseline-from costs.json] [--compare before.json]
        [--memory-max 8G] [--timeout 900]

`--baseline-from` reuses the halo2_hw numbers of an earlier run (they only change
when the baselines change). `--compare` prints a before/after table against an
earlier output of this script. Every Zinnia compile runs in its own child process
under the memory cap (`eval-redo/tools/memrun.sh` when present), so a case that
exhausts memory is recorded as such instead of stopping the run.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarking.harness import paths

HALO2_BIN = paths.BASELINES / "halo2_hw" / "target" / "release" / "halo2_hw"
MEMRUN = Path("/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh")

#: IR operations worth listing next to a case's cost (the ones with gadgets).
CENSUS_OPS = frozenset({
    "MulF", "ProductSumF", "DivF", "FloorDivI", "FloorDivIPinned", "ModI", "FloorDivF", "ModF",
    "SinF", "CosF", "ArcSinF", "SqrtF", "PowI", "PowF", "PowFI", "IntCast", "DivI", "InvI",
    "LessThanI", "LessThanF", "LessThanOrEqualI", "LessThanOrEqualF", "GreaterThanI",
    "GreaterThanF", "GreaterThanOrEqualI", "GreaterThanOrEqualF", "LessThanIBounded",
    "AbsI", "AbsF", "SignI", "SignF", "ReadMemory", "WriteMemory", "BitAndI", "BitOrI",
    "BitXorI", "EqualI", "EqualF", "NotEqualI", "NotEqualF", "SelectI", "SelectF", "SelectB",
})


def case_id(text: str) -> str:
    return text if text.startswith("ds1000_case") else f"ds1000_case{int(text):04d}"


def case_entries() -> dict[str, dict]:
    raw = json.loads(paths.PERF_CASE_SET.read_text())
    return {entry["case"]: entry for entry in raw["cases"]}


def baseline_cases(entries: dict[str, dict]) -> list[str]:
    """Cases with a halo2_hw baseline, in case order."""
    return sorted(c for c in entries if (paths.DS1000_PERF / c / "target.rs").exists())


def fixture_path(case: str, entries: dict[str, dict]) -> Path | None:
    """The case's fixture, or that of the case it duplicates (same signature)."""
    own = paths.DS1000_PERF / case / "fixture.json"
    if own.exists():
        return own
    original = entries.get(case, {}).get("duplicate_of")
    if original:
        theirs = paths.DS1000_PERF / original / "fixture.json"
        if theirs.exists():
            return theirs
    return None


# ------------------------------------------------------------------ Zinnia side


def zinnia_one(case: str) -> dict:
    """Compile one kernel and measure it (runs inside the child process)."""
    from benchmarking.harness.kernels import compile_kernel
    from zinnia._zinnia_core import halo2_base_cell_stats

    started = time.perf_counter()
    compiled = compile_kernel(paths.DS1000_PERF / case / "kernel.py", "kernel")
    compiled_s = time.perf_counter() - started
    census = collections.Counter()
    for stmt in compiled.get_ir_stmts():
        name = str(stmt["ir_instance"]["__class__"]).removesuffix("IR")
        if name in CENSUS_OPS:
            census[name] += 1
    stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
    stats["ops"] = dict(sorted(census.items()))
    stats["ir_stmts"] = len(compiled.get_ir_stmts())
    stats["compile_s"] = round(compiled_s, 3)
    stats["measure_s"] = round(time.perf_counter() - started - compiled_s, 3)
    return stats


def zinnia_child(case: str, memory_max: str | None, timeout: float) -> dict:
    cmd = [sys.executable, "-m", "benchmarking.harness.costs_halo2", "--zinnia-one", case]
    env = dict(os.environ)
    if memory_max and MEMRUN.exists():
        cmd = [str(MEMRUN), *cmd]
        env["MEMRUN_MAX"] = memory_max
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=paths.REPO_ROOT, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    reason = "killed (out of memory?)" if proc.returncode in (-9, 137) else f"exit {proc.returncode}"
    return {"error": f"{reason}: {' | '.join(tail)}"[:600]}


# ---------------------------------------------------------------- baseline side


def halo2_hw_stats(case: str, fixture: Path | None, binary: Path, timeout: float) -> dict:
    if fixture is None:
        return {"error": "no fixture"}
    cmd = [str(binary), "stats", "--case", str(paths.DS1000_PERF / case),
           "--fixture", str(fixture), "--input", "A"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            out = json.loads(line)
            if not out.get("ok"):
                return {"error": out.get("error", "not ok")}
            cells = out["cells"]
            return {"k": out["k"], "lookup_bits": out["lookup_bits"], **cells}
    return {"error": f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}"}


# --------------------------------------------------------------------- analysis


def committed(record: dict | None) -> int | None:
    if not record or "error" in record:
        return None
    return record["advice"] + record["lookup"]


def ratio(row: dict) -> float | None:
    z, h = committed(row.get("zinnia")), committed(row.get("halo2_hw"))
    if z is None or h is None or h == 0:
        return None
    return z / h


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summary(rows: dict[str, dict]) -> dict:
    perf = [ratio(r) for r in rows.values() if r["status"] == "perf"]
    every = [ratio(r) for r in rows.values()]
    return {
        "geomean_ratio_perf": geomean([r for r in perf if r is not None]),
        "geomean_ratio_all": geomean([r for r in every if r is not None]),
        "measured_perf": sum(r is not None for r in perf),
        "measured_all": sum(r is not None for r in every),
        "zinnia_cheaper_perf": sum(r is not None and r < 1 for r in perf),
        "zinnia_failed": sorted(c for c, r in rows.items() if "error" in (r.get("zinnia") or {})),
    }


def fmt_cells(record: dict | None) -> str:
    if not record:
        return "-"
    if "error" in record:
        return "error"
    return f"{record['advice']} + {record['lookup']}"


def fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def table(rows: dict[str, dict]) -> str:
    """Markdown table sorted by ratio, worst first."""
    lines = ["| case | status | Zinnia advice + lookup | k | halo2_hw advice + lookup | k | ratio |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(rows, key=lambda c: -(ratio(rows[c]) or float("inf")))
    for case in order:
        row = rows[case]
        z, h = row.get("zinnia") or {}, row.get("halo2_hw") or {}
        lines.append(f"| {case[-4:]} | {row['status']} | {fmt_cells(z)} | {z.get('k', '-')} | "
                     f"{fmt_cells(h)} | {h.get('k', '-')} | {fmt_ratio(ratio(row))} |")
    return "\n".join(lines)


def comparison(before: dict[str, dict], after: dict[str, dict]) -> str:
    """Markdown before/after table of the Zinnia side, sorted by after-ratio."""
    lines = ["| case | status | Zinnia before | k | Zinnia after | k | halo2_hw | k | "
             "ratio before | ratio after |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(after, key=lambda c: -(ratio(after[c]) or float("inf")))
    for case in order:
        a = after[case]
        b = before.get(case, {})
        zb, za, h = b.get("zinnia") or {}, a.get("zinnia") or {}, a.get("halo2_hw") or {}
        lines.append(
            f"| {case[-4:]} | {a['status']} | {fmt_cells(zb)} | {zb.get('k', '-')} | "
            f"{fmt_cells(za)} | {za.get('k', '-')} | {fmt_cells(h)} | {h.get('k', '-')} | "
            f"{fmt_ratio(ratio(b) if b else None)} | {fmt_ratio(ratio(a))} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma-separated case numbers (default: every case with a target.rs)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--halo2-bin", type=Path, default=HALO2_BIN)
    ap.add_argument("--baseline-from", type=Path, help="reuse halo2_hw numbers from an earlier output")
    ap.add_argument("--compare", type=Path, help="earlier output to print a before/after table against")
    ap.add_argument("--memory-max", default="8G", help="per-case memory cap ('none' to disable)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--zinnia-one", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.zinnia_one:
        print(json.dumps(zinnia_one(args.zinnia_one)), flush=True)
        return 0

    entries = case_entries()
    cases = [case_id(c) for c in args.cases.split(",")] if args.cases else baseline_cases(entries)
    reused = json.loads(args.baseline_from.read_text())["cases"] if args.baseline_from else {}
    memory_max = None if args.memory_max.lower() == "none" else args.memory_max

    rows: dict[str, dict] = {}
    for case in cases:
        fixture = fixture_path(case, entries)
        if case in reused and "halo2_hw" in reused[case]:
            hw = reused[case]["halo2_hw"]
        else:
            hw = halo2_hw_stats(case, fixture, args.halo2_bin, args.timeout)
        zinnia = zinnia_child(case, memory_max, args.timeout)
        rows[case] = {"status": entries.get(case, {}).get("status", "unknown"),
                      "zinnia": zinnia, "halo2_hw": hw}
        print(f"{case}  zinnia={fmt_cells(zinnia)} k={zinnia.get('k')}  "
              f"halo2_hw={fmt_cells(hw)} k={hw.get('k')}  ratio={fmt_ratio(ratio(rows[case]))}"
              + (f"  {zinnia['error']}" if "error" in zinnia else ""), flush=True)
        if args.out:
            args.out.write_text(json.dumps({"cases": rows, "summary": summary(rows)}, indent=1))

    result = {"cases": rows, "summary": summary(rows)}
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    print()
    if args.compare:
        before = json.loads(args.compare.read_text())
        print(comparison(before["cases"], rows))
        print()
        print("before:", json.dumps(before["summary"]))
    else:
        print(table(rows))
        print()
    print("after:" if args.compare else "summary:", json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
