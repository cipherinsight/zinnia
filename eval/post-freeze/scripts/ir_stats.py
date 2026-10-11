"""IR digest, statement count and assertion counts of every included perf case, both targets.

Usage (from a worktree): .venv/bin/python <this> OUT.json [CASE_SET]
Ablation switches are read from the environment by the compiler itself.
"""
import collections, json, os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
from benchmarking.harness.case_sets import load_perf_cases
from benchmarking.harness.adapters.zinnia_common import compile_case, ir_sha256

DOMAIN = ("GreaterThanOrEqualIIR", "LessThanOrEqualIIR")


def stats(compiled):
    stmts = compiled.get_ir_stmts()
    by_id = {s["stmt_id"]: s for s in stmts}
    kinds = collections.Counter(s["ir_instance"]["__class__"] for s in stmts)
    asserts = {k: v for k, v in kinds.items() if k.startswith("Assert")}
    reads = {s["stmt_id"] for s in stmts if s["ir_instance"]["__class__"] in ("ReadIntegerIR", "ReadFloatIR")}
    domain = 0
    for s in stmts:
        if s["ir_instance"]["__class__"] == "AssertIR":
            c = by_id.get(s["arguments"][0])
            if c and c["ir_instance"]["__class__"] in DOMAIN and c["arguments"][0] in reads:
                domain += 1
        elif s["ir_instance"]["__class__"] == "AssertBitIR" and s["arguments"][0] in reads:
            domain += 1
    return {"sha": ir_sha256(stmts), "stmts": len(stmts), "asserts": sum(asserts.values()),
            "domain_asserts": domain, "by_kind": asserts}


def main(out, case_set="benchmarking/datasets/case_sets/ds1000_perf.json"):
    res = {}
    for case in load_perf_cases(Path(case_set), None, None):
        if not case.included:
            continue
        row = {}
        for backend in ("halo2", "noir"):
            try:
                row[backend] = stats(compile_case(case, backend))
            except BaseException as exc:
                row[backend] = {"error": f"{exc}"[:300]}
        res[case.case] = row
        print(case.case, {b: (r.get("asserts"), r.get("sha", r.get("error", ""))[:12]) for b, r in row.items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)


main(*sys.argv[1:])
