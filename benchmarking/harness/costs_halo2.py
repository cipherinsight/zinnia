"""Deterministic halo2-base cost table: Zinnia against the handwritten halo2_hw baselines.

For every DS-1000 case that has a handwritten halo2-base baseline (`target.rs`):

* **Zinnia**: compile `kernel.py`, let the halo2-base backend's `estimate_params`
  choose `(k, lookup_bits)`, and count the circuit's advice, lookup-advice and
  fixed cells at that lookup width (a witness-free synthesis; cell counts do not
  depend on the inputs).
* **halo2_hw**: `halo2_hw stats` with input `A` of the case's fixture (the
  baseline's own copy of the same sizing rule).

**Cells** are advice + lookup cells (the cells the prover commits to). Both sides
pick `k` by the same rule, `lookup_bits = min(8, k - 1)`, so when the two `k`s
differ the two circuits are laid out at different **lookup widths**: every range
check is split into `lookup_bits`-wide limbs, and a circuit at a narrower width
has more limbs (more cells) for the same constraints. Cells at different widths
are not comparable: a case can look costlier only because its `k`, and with it
its lookup width, is smaller than the other side's.

Each row therefore records, per side, the cells at its own chosen `k` (and that
`k` and lookup width), and both sides' cells at the **common lookup width**, the
larger of the two sides' `lookup_bits` (the side whose width is narrower is
re-synthesized at the wider one: Zinnia through `halo2_base_cell_stats` with
explicit parameters, the baseline through `halo2_hw stats --k --lookup-bits`).
**The comparison ratio is Zinnia / halo2_hw at the common lookup width**; the
ratio at each side's own `k` is listed next to it (`ratio_own_k`). When both sides
choose the same `k` the two ratios are equal. The summary gives the geometric
mean of the common-width ratio over the `perf` cases and over every case
measured (functional duplicates included). Proving time is measured by the
harness, not here.

Usage:

    python -m benchmarking.harness.costs_halo2 [--cases 0309,0310] [--out costs.json]
        [--halo2-bin PATH] [--baseline-from costs.json] [--compare before.json]
        [--memory-max 8G] [--timeout 900]

`--profile 0450` prints, for one case, the advice and lookup cells per IR operation
type (at lookup width 8), which shows where a circuit's cells come from.

`--check` proves every selected case instead of measuring it: key generation on
input A with `estimate_params` (retried at the next `k` if halo2-base cannot lay
the circuit out), a proof on input B, verification, rejection of a proof whose
first public value is changed, and the outputs compared with the NumPy reference
position by position (the harness's decoder and tolerance) and with the mock
backend's public values.

`--baseline-from` reuses the halo2_hw numbers of an earlier run (they only change
when the baselines change; the baseline's cells at a common width it did not
choose are measured when that output lacks them). `--compare` prints a
before/after table against an earlier output of this script. Every Zinnia compile
runs in its own child process under the memory cap (`eval-redo/tools/memrun.sh`
when present), so a case that exhausts memory is recorded as such instead of
stopping the run.
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


def zinnia_one(case: str, other_lookup_bits: int | None = None) -> dict:
    """Compile one kernel and measure it (runs inside the child process).

    When `other_lookup_bits` (the baseline's width) is wider than the width
    Zinnia chose, the record also carries `at_common`: the cells at that width."""
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
    if other_lookup_bits is not None and other_lookup_bits > stats["lookup_bits"]:
        params = {"k": stats["k"], "precision_bits": stats["precision_bits"],
                  "lookup_bits": other_lookup_bits}
        wider = json.loads(halo2_base_cell_stats(compiled._compiled_ir, json.dumps(params)))
        stats["at_common"] = {key: wider[key] for key in ("lookup_bits", "advice", "lookup")}
    stats["ops"] = dict(sorted(census.items()))
    stats["ir_stmts"] = len(compiled.get_ir_stmts())
    stats["compile_s"] = round(compiled_s, 3)
    stats["measure_s"] = round(time.perf_counter() - started - compiled_s, 3)
    return stats


def zinnia_profile(case: str, lookup_bits: int = 8) -> list[tuple[str, int, int, int]]:
    """`(IR operation, statements, advice cells, lookup cells)`, most advice first."""
    from benchmarking.harness.kernels import compile_kernel
    from zinnia._zinnia_core import halo2_base_cell_profile

    compiled = compile_kernel(paths.DS1000_PERF / case / "kernel.py", "kernel")
    op_of = {stmt["stmt_id"]: str(stmt["ir_instance"]["__class__"]).removesuffix("IR")
             for stmt in compiled.get_ir_stmts()}
    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for stmt_id, advice, lookup in json.loads(halo2_base_cell_profile(compiled._compiled_ir, lookup_bits)):
        entry = totals[op_of.get(stmt_id, "?")]
        entry[0] += 1
        entry[1] += advice
        entry[2] += lookup
    return sorted(((op, *v) for op, v in totals.items()), key=lambda row: -row[2])


def mock_publics_in_order(labelled: list[str]) -> list[int]:
    """The mock's raw public values (`public_<kind>_<statement id>=<raw>`, listed
    in hash order) in statement order, which is the order halo2-base exposes."""
    def key(entry: str) -> int:
        return int(entry.split("=", 1)[0].rsplit("_", 1)[1])
    return [int(entry.rsplit("=", 1)[1]) for entry in sorted(labelled, key=key)]


def zinnia_check(case: str) -> dict:
    """Prove input B with keys from input A, verify, tamper, compare (child process)."""
    from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
    from benchmarking.harness.case_sets import parse_tolerance
    from benchmarking.harness.fixtures import PROVE_INPUT, load_fixture
    from benchmarking.harness.kernels import compile_kernel
    from benchmarking.harness.outputs import (check_outputs, field_from_le_hex, field_to_le_hex,
                                              field_to_signed, reference_leaves)
    from benchmarking.harness.reference import run_kernel_as_numpy

    entries = case_entries()
    kernel = paths.DS1000_PERF / case / "kernel.py"
    fixture = load_fixture(fixture_path(case, entries))
    compiled = compile_kernel(kernel, "kernel")
    args = fixture.arguments(PROVE_INPUT)
    started = time.perf_counter()
    keys = compiled.setup()
    setup_s = time.perf_counter() - started
    proof = compiled.prove_with_keys(keys, *args)
    verified = compiled.verify_with_keys(keys, proof)
    tampered = dict(proof)
    if proof["public_values"]:
        first = field_from_le_hex(proof["public_values"][0])
        tampered["public_values"] = [field_to_le_hex(first + 1), *proof["public_values"][1:]]
        tamper_rejected = not compiled.verify_with_keys(keys, tampered)
    else:
        tamper_rejected = None
    fields = [field_from_le_hex(v) for v in proof["public_values"]]
    mock = compiled.prove(*args, backend="mock")
    mock_raw = mock_publics_in_order(mock.public_values)
    halo2_raw = [field_to_signed(v) for v in fields]
    reference = run_kernel_as_numpy(kernel, "kernel", fixture.arguments(PROVE_INPUT))
    leaves, _ = reference_leaves(reference)
    tolerance = parse_tolerance(entries[case].get("float_tolerance"), "case set")
    numpy = check_outputs(leaves, zinnia_publics(fields, describe_compiled(compiled)), tolerance)
    return {
        "k": keys.k, "lookup_bits": keys.lookup_bits, "setup_s": round(setup_s, 3),
        "prove_s": round(proof["prove_seconds"], 3), "verified": verified,
        "tamper_rejected": tamper_rejected, "publics": len(fields),
        "mock_equal": halo2_raw == mock_raw,
        "numpy": numpy.status, "numpy_detail": numpy.detail,
    }


def known_mismatches() -> set[str]:
    """Perf cases where the case set records that Zinnia computes another function."""
    raw = json.loads(paths.PERF_CASE_SET.read_text())
    return set(raw.get("zinnia_output_mismatches", {}).get("cases", []))


def check_passed(case: str, record: dict, known: set[str]) -> bool:
    # A NumPy mismatch always fails: the case set's `zinnia_output_mismatches` comes from the generator's
    # probe and is only reported, never used as an exemption (it would hide a regression).
    numpy_ok = record.get("numpy") in ("match", "unchecked")
    return (record.get("verified") is True and record.get("tamper_rejected") is not False
            and record.get("mock_equal") is True and numpy_ok)


def zinnia_child(case: str, memory_max: str | None, timeout: float, mode: str = "--zinnia-one",
                 other_lookup_bits: int | None = None) -> dict:
    cmd = [sys.executable, "-m", "benchmarking.harness.costs_halo2", mode, case]
    if other_lookup_bits is not None:
        cmd += ["--other-lookup-bits", str(other_lookup_bits)]
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


def halo2_hw_stats(case: str, fixture: Path | None, binary: Path, timeout: float,
                   lookup_bits: int | None = None) -> dict:
    """The baseline's cells at its own chosen size, or at `lookup_bits` if given."""
    if fixture is None:
        return {"error": "no fixture"}
    cmd = [str(binary), "stats", "--case", str(paths.DS1000_PERF / case),
           "--fixture", str(fixture), "--input", "A"]
    if lookup_bits is not None:
        # `--lookup-bits` needs a `k`; the cell counts depend on the width only.
        cmd += ["--k", str(lookup_bits + 1), "--lookup-bits", str(lookup_bits)]
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


def ratio_own_k(row: dict) -> float | None:
    """Zinnia / halo2_hw cells, each side at its own chosen `k` and lookup width."""
    z, h = committed(row.get("zinnia")), committed(row.get("halo2_hw"))
    if z is None or h is None or h == 0:
        return None
    return z / h


def ratio(row: dict) -> float | None:
    """The comparison ratio: Zinnia / halo2_hw cells at the common lookup width
    (the larger of the two sides' `lookup_bits`). Rows written before this field
    existed fall back to the own-`k` ratio."""
    common = row.get("common")
    if not common:
        return ratio_own_k(row)
    z, h = committed(common.get("zinnia")), committed(common.get("halo2_hw"))
    if z is None or h is None or h == 0:
        return None
    return z / h


def common_width(zinnia: dict, hw: dict, case: str, fixture: Path | None, binary: Path,
                 timeout: float, reused: dict | None) -> dict | None:
    """Both sides' cells at the larger of their two lookup widths."""
    if "error" in zinnia or "error" in hw:
        return None
    width = max(zinnia["lookup_bits"], hw["lookup_bits"])
    if width == zinnia["lookup_bits"]:
        z = {"advice": zinnia["advice"], "lookup": zinnia["lookup"]}
    else:
        z = {key: zinnia["at_common"][key] for key in ("advice", "lookup")}
    if width == hw["lookup_bits"]:
        h = {"advice": hw["advice"], "lookup": hw["lookup"]}
    elif reused and reused.get("lookup_bits") == width and "halo2_hw" in reused:
        h = reused["halo2_hw"]
    else:
        wider = halo2_hw_stats(case, fixture, binary, timeout, lookup_bits=width)
        if "error" in wider:
            return {"lookup_bits": width, "zinnia": z, "halo2_hw": wider}
        h = {"advice": wider["advice"], "lookup": wider["lookup"]}
    return {"lookup_bits": width, "zinnia": z, "halo2_hw": h}


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
        "zinnia_costlier_perf": sum(r is not None and r > 1 for r in perf),
        "geomean_ratio_own_k_perf": geomean([r for r in (ratio_own_k(x) for x in rows.values()
                                                          if x["status"] == "perf") if r is not None]),
        "ratio": "Zinnia / halo2_hw cells at the common lookup width (max of both sides' lookup_bits)",
        "zinnia_failed": sorted(c for c, r in rows.items() if "error" in (r.get("zinnia") or {})),
    }


def fmt_cells(record: dict | None) -> str:
    if not record:
        return "-"
    if "error" in record:
        return "error"
    return f"{record['advice']} + {record['lookup']}"


def fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def fmt_k(record: dict | None) -> str:
    if not record or "error" in record:
        return "-"
    return f"{record['k']}/{record['lookup_bits']}"


def table(rows: dict[str, dict]) -> str:
    """Markdown table sorted by the common-width ratio, worst first."""
    lines = ["| case | status | Zinnia advice + lookup | k/lb | halo2_hw advice + lookup | k/lb | "
             "ratio own k | common lb | Zinnia at common | halo2_hw at common | ratio |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(rows, key=lambda c: -(ratio(rows[c]) or float("inf")))
    for case in order:
        row = rows[case]
        z, h = row.get("zinnia") or {}, row.get("halo2_hw") or {}
        common = row.get("common") or {}
        lines.append(f"| {case[-4:]} | {row['status']} | {fmt_cells(z)} | {fmt_k(z)} | "
                     f"{fmt_cells(h)} | {fmt_k(h)} | {fmt_ratio(ratio_own_k(row))} | "
                     f"{common.get('lookup_bits', '-')} | {fmt_cells(common.get('zinnia'))} | "
                     f"{fmt_cells(common.get('halo2_hw'))} | {fmt_ratio(ratio(row))} |")
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


def run_checks(cases: list[str], entries: dict[str, dict], args: argparse.Namespace,
               memory_max: str | None) -> int:
    results = {}
    known = known_mismatches()
    for case in cases:
        record = zinnia_child(case, memory_max, args.timeout, mode="--zinnia-check")
        record["status"] = entries.get(case, {}).get("status", "unknown")
        results[case] = record
        verdict = "ok" if check_passed(case, record, known) else "FAIL"
        print(f"{case}  {verdict}  " + json.dumps({k: v for k, v in record.items() if k != "status"}),
              flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=1))
    failed = sorted(c for c, r in results.items() if not check_passed(c, r, known))
    numpy_checked = sum(r.get("numpy") == "match" for r in results.values())
    print(f"\n{len(results) - len(failed)} of {len(results)} pass; {numpy_checked} compared with NumPy "
          f"position by position; listed as mismatches by the case set's probe (stale if they pass): "
          f"{sorted(c for c in results if c in known)}; failed: {failed}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", help="comma-separated case numbers (default: every case with a target.rs)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--halo2-bin", type=Path, default=HALO2_BIN)
    ap.add_argument("--baseline-from", type=Path, help="reuse halo2_hw numbers from an earlier output")
    ap.add_argument("--compare", type=Path, help="earlier output to print a before/after table against")
    ap.add_argument("--memory-max", default="8G", help="per-case memory cap ('none' to disable)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--profile", help="print the cells per IR operation type of one case")
    ap.add_argument("--check", action="store_true", help="prove, verify and output-check instead")
    ap.add_argument("--zinnia-one", help=argparse.SUPPRESS)
    ap.add_argument("--other-lookup-bits", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--zinnia-check", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.zinnia_check:
        print(json.dumps(zinnia_check(args.zinnia_check)), flush=True)
        return 0

    if args.profile:
        rows = zinnia_profile(case_id(args.profile))
        print(f"{'operation':24} {'stmts':>6} {'advice':>9} {'lookup':>8}")
        for op, count, advice, lookup in rows:
            print(f"{op:24} {count:6d} {advice:9d} {lookup:8d}")
        print(f"{'total':24} {sum(r[1] for r in rows):6d} {sum(r[2] for r in rows):9d} "
              f"{sum(r[3] for r in rows):8d}")
        return 0

    if args.zinnia_one:
        print(json.dumps(zinnia_one(args.zinnia_one, args.other_lookup_bits)), flush=True)
        return 0

    entries = case_entries()
    cases = [case_id(c) for c in args.cases.split(",")] if args.cases else baseline_cases(entries)
    if args.check:
        return run_checks(cases, entries, args, memory_max=None if args.memory_max.lower() == "none"
                          else args.memory_max)
    reused = json.loads(args.baseline_from.read_text())["cases"] if args.baseline_from else {}
    memory_max = None if args.memory_max.lower() == "none" else args.memory_max

    rows: dict[str, dict] = {}
    for case in cases:
        fixture = fixture_path(case, entries)
        if case in reused and "halo2_hw" in reused[case]:
            hw = reused[case]["halo2_hw"]
        else:
            hw = halo2_hw_stats(case, fixture, args.halo2_bin, args.timeout)
        zinnia = zinnia_child(case, memory_max, args.timeout, other_lookup_bits=hw.get("lookup_bits"))
        common = common_width(zinnia, hw, case, fixture, args.halo2_bin, args.timeout,
                              (reused.get(case) or {}).get("common"))
        rows[case] = {"status": entries.get(case, {}).get("status", "unknown"),
                      "zinnia": zinnia, "halo2_hw": hw, "common": common}
        print(f"{case}  zinnia={fmt_cells(zinnia)} k={fmt_k(zinnia)}  "
              f"halo2_hw={fmt_cells(hw)} k={fmt_k(hw)}  ratio_own_k={fmt_ratio(ratio_own_k(rows[case]))}  "
              f"common lb={(common or {}).get('lookup_bits')} ratio={fmt_ratio(ratio(rows[case]))}"
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
