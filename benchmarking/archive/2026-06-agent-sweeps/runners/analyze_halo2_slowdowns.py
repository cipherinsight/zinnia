"""Classify remaining Zinnia-vs-HW Halo2 slowdown causes.

The headline benchmark `prove_s` is the Rust/Halo2 `create_proof` window.
That excludes Python compilation/lowering, witness JSON parsing, keygen, and
artifact assembly. It still includes Halo2 circuit synthesis because synthesis
runs inside `create_proof`.

This script separates true proving/constraint slowdowns from cases where the
measured gap is dominated by synthesis/public-instance overhead or tiny-circuit
fixed costs. It overlays targeted Zinnia result files on top of a full sweep so
the latest local patches can be included without rerunning the whole suite.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).parent
ROOT = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))


ABANDONED_BASELINE_CASES = {
    "0298",
    "0383",
    "0388",
    "0389",
    "0390",
    "0391",
    "0392",
    "0393",
    "0467",
    "0468",
}


def load_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data.get("results", data if isinstance(data, list) else [])
    return {r["case"]: r for r in rows}


def median_hw_prove(row: dict) -> float | None:
    if row.get("prove_s") is not None:
        return row["prove_s"]
    times = row.get("prove_times_s") or []
    return statistics.median(times) if times else None


def compile_ir_counts(case: str) -> Counter:
    from zinnia.api.zk_circuit import ZKCircuit

    kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"case{case}_slowdown", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    compiled = ZKCircuit.from_method(mod.kernel).compile()
    return Counter(stmt["ir_instance"]["__class__"] for stmt in compiled.get_ir_stmts())


def classify(cnt: Counter, z_prove_s: float, z_k: int | None) -> tuple[str, bool]:
    total = sum(cnt.values())
    expose = cnt["ExposePublicFIR"] + cnt["ExposePublicIIR"]
    reads = sum(v for k, v in cnt.items() if k.startswith("Read"))
    mulf = cnt["MulFIR"]
    product_sum = cnt["ProductSumFIR"]
    selects = sum(v for k, v in cnt.items() if "Select" in k)
    bit_asserts = cnt["AssertBitIR"]
    comparisons = sum(
        cnt[k]
        for k in (
            "LtIIR",
            "LteIIR",
            "GtIIR",
            "GteIIR",
            "EqIIR",
            "NotEqIIR",
            "LtFIR",
            "LteFIR",
            "GtFIR",
            "GteFIR",
            "EqFIR",
            "NotEqFIR",
        )
    )
    addf = cnt["AddFIR"] + cnt["SubFIR"]

    # Exclude from high-impact proving-cause analysis: these are dominated by
    # Halo2 synthesis/public-instance assignment or fixed framework overhead,
    # not by arithmetic proving constraints.
    if total and reads + expose >= total - 5 and expose >= 1000:
        return "layout/public-only synthesis dominated", True
    if z_k is not None and z_k <= 6 and z_prove_s < 0.30:
        return "tiny-circuit fixed overhead", True

    if product_sum or mulf >= 100:
        return "fixed-point mul/rescale range checks", False
    if selects >= 50 or bit_asserts >= 50:
        return "select/gather masking + bit asserts", False
    if comparisons >= 50:
        return "comparison/argmin/max bit gadgets", False
    if addf >= 20:
        return "small float add/full-config overhead", False
    if expose >= 100:
        return "public output overhead", False
    return "mixed/other", False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zinnia", default=str(THIS_DIR / "codex_postfix_20260607_zinnia_halo2.json"))
    ap.add_argument("--hw", default=str(THIS_DIR / "codex_postfix_20260607_hw_halo2.json"))
    ap.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="additional Zinnia result JSON; later files override earlier rows",
    )
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    zinnia = load_results(Path(args.zinnia))
    for overlay in args.overlay:
        zinnia.update(load_results(Path(overlay)))
    hw = load_results(Path(args.hw))

    rows = []
    for case, zr in zinnia.items():
        if case in ABANDONED_BASELINE_CASES:
            continue
        hr = hw.get(case)
        z_prove = zr.get("prove_s")
        h_prove = median_hw_prove(hr or {})
        if (
            not hr
            or z_prove is None
            or h_prove is None
            or zr.get("error")
            or hr.get("error")
            or not zr.get("verify_ok")
            or not hr.get("verify_ok")
        ):
            continue
        rows.append(
            {
                "case": case,
                "ratio": z_prove / h_prove,
                "delta": z_prove - h_prove,
                "z_prove": z_prove,
                "h_prove": h_prove,
                "z_k": zr.get("min_k"),
                "h_k": hr.get("min_k"),
            }
        )

    rows.sort(key=lambda r: r["ratio"], reverse=True)

    cause_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    included = []
    excluded = []

    for row in rows:
        if row["ratio"] <= 1.0:
            continue
        try:
            counts = compile_ir_counts(row["case"])
            cause, exclude = classify(counts, row["z_prove"], row["z_k"])
        except Exception as exc:  # noqa: BLE001
            counts = Counter({"COMPILE_PROFILE_ERROR": 1})
            cause, exclude = (f"profile error: {type(exc).__name__}", False)
        row["cause"] = cause
        row["top_ir"] = counts.most_common(5)
        if exclude:
            excluded_counts[cause] += 1
            excluded.append(row)
        else:
            cause_counts[cause] += 1
            included.append(row)

    print(f"Both verified, non-abandoned cases: {len(rows)}")
    print(f"Zinnia slower before measurement exclusions: {len(included) + len(excluded)}")
    print(f"Excluded synthesis/fixed-overhead cases: {len(excluded)}")
    print(f"High-impact slower cases after exclusions: {len(included)}")
    print()

    print("Excluded categories:")
    for cause, n in excluded_counts.most_common():
        print(f"  {n:3d}  {cause}")
    print()

    print("Remaining high-impact categories:")
    for cause, n in cause_counts.most_common():
        print(f"  {n:3d}  {cause}")
    print()

    by_cause: dict[str, list[dict]] = defaultdict(list)
    for row in included:
        by_cause[row["cause"]].append(row)

    print("Top slower cases after exclusions:")
    for row in included[: args.top]:
        print(
            f"  {row['case']} ratio={row['ratio']:.2f} "
            f"delta={row['delta']:.3f}s z={row['z_prove']:.3f}s hw={row['h_prove']:.3f}s "
            f"k={row['z_k']}/{row['h_k']} cause={row['cause']} top_ir={row['top_ir']}"
        )

    print()
    print("Largest absolute slowdowns after exclusions:")
    for row in sorted(included, key=lambda r: r["delta"], reverse=True)[: args.top]:
        print(
            f"  {row['case']} delta={row['delta']:.3f}s "
            f"ratio={row['ratio']:.2f} z={row['z_prove']:.3f}s hw={row['h_prove']:.3f}s "
            f"k={row['z_k']}/{row['h_k']} cause={row['cause']} top_ir={row['top_ir']}"
        )


if __name__ == "__main__":
    main()
