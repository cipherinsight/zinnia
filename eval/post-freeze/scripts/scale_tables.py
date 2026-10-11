"""Markdown tables of the post-freeze scaling series.

Usage: python3 scale_tables.py DATA_SCALE_DIR [--ref LOCAL_REF_DIR]
  DATA_SCALE_DIR: challenge/scale_<family>.jsonl, hw/scale_zinnia.jsonl, hw/scale_hw.jsonl,
                  memory_cells.jsonl, memory_auto_timed.jsonl
  --ref: a directory of the same layout with cells-only runs (e.g. the local run at the
         same commit); every cell / gate count that differs is listed.
Times: median of 3 proofs. "create_proof" is the prover call alone (the baseline's prove_s);
"wall" adds Zinnia's witness generation.
"""
import json
import sys
from pathlib import Path


def rows(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def f(x, d=3):
    return "-" if x is None else (f"{x:,}" if isinstance(x, int) else f"{x:.{d}f}")


def ok(r):
    if "verified" not in r:
        return "-"
    good = r.get("verified") and r.get("tamper_rejected") and (r.get("max_abs_error", 0) or 0) < 1e-6
    return "ok" if good else f"FAIL(v={r.get('verified')},t={r.get('tamper_rejected')},e={r.get('max_abs_error')})"


def challenge(d):
    labels = {"argsort": ("gp", "routed"), "matmul": ("freivalds", "direct"), "convolve": ("identity", "direct")}
    for fam, (a, b) in labels.items():
        rs = rows(d / "challenge" / f"scale_{fam}.jsonl")
        if not rs:
            continue
        by = {(r["n"], r["label"]): r for r in rs}
        print(f"\n**{fam}** (halo2-base; cells (k); prove s = median create_proof / wall)\n")
        print(f"| N | auto picks | auto cells (k) | {a} cells (k) | {b} cells (k) | {a}/{b} cells | prove s auto | prove s {a} | prove s {b} | {a}/{b} time | checks |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for n in sorted({r["n"] for r in rs}):
            au, x, y = by.get((n, "auto"), {}), by.get((n, a), {}), by.get((n, b), {})
            pick = a if au.get("cells") == x.get("cells") and au.get("k") == x.get("k") else (
                b if au.get("cells") == y.get("cells") and au.get("k") == y.get("k") else "?")
            cell = lambda r: f"{f(r.get('cells'))} (k {r.get('k')})" if r.get("cells") else r.get("error", "-")[:60]
            t = lambda r: f"{f(r.get('create_proof_median_s'))} / {f(r.get('prove_median_s'))}"
            ratio_c = x["cells"] / y["cells"] if x.get("cells") and y.get("cells") else None
            ratio_t = (x["create_proof_median_s"] / y["create_proof_median_s"]
                       if x.get("create_proof_median_s") and y.get("create_proof_median_s") else None)
            checks = ",".join(sorted({ok(r) for r in (au, x, y)}))
            print(f"| {n} | {pick} | {cell(au)} | {cell(x)} | {cell(y)} | {f(ratio_c, 2)} | {t(au)} | {t(x)} | {t(y)} | {f(ratio_t, 2)} | {checks} |")


def hw(d):
    zs = {r["case"]: r for r in rows(d / "hw" / "scale_zinnia.jsonl")}
    hs = {r["case"]: r for r in rows(d / "hw" / "scale_hw.jsonl")}
    if not zs and not hs:
        return
    print("\n**Zinnia (own choice) vs handwritten halo2_hw, scaled perf cases** (cells at each side's own k; "
          "prove s = median of 3 create_proof; Zinnia wall incl. witness in brackets)\n")
    print("| case | N | Zinnia cells (k) | baseline cells (k) | cells ratio | Zinnia prove s [wall] | baseline prove s | time ratio | Zinnia checks | baseline checks |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for case in sorted(set(zs) | set(hs)):
        z, h = zs.get(case, {}), hs.get(case, {})
        base, n = case[len("ds1000_case"):][:4], int(case[-4:])
        rc = z["cells"] / h["cells"] if z.get("cells") and h.get("cells") else None
        rt = (z["create_proof_median_s"] / h["create_proof_median_s"]
              if z.get("create_proof_median_s") and h.get("create_proof_median_s") else None)
        hchk = "-" if "verified" not in h else ("ok" if h.get("verified") and h.get("tamper_rejected") and h.get("prove_ok") else
                                                f"FAIL(p={h.get('prove_ok')},v={h.get('verified')},t={h.get('tamper_rejected')})")
        print(f"| {base} | {n} | {f(z.get('cells'))} (k {z.get('k')}) | {f(h.get('cells'))} (k {h.get('k')}) | {f(rc, 3)} | "
              f"{f(z.get('create_proof_median_s'))} [{f(z.get('prove_median_s'))}] | {f(h.get('create_proof_median_s'))} | {f(rt, 2)} | {ok(z)} | {hchk} |")


def memory(d):
    cells = rows(d / "memory_cells.jsonl")
    timed = {(r["kernel"], r["n"]): r for r in rows(d / "memory_auto_timed.jsonl")}
    if not cells and not timed:
        return
    forces = ["chain", "indicators", "memory", "bits", "table"]
    by = {(r["kernel"], r["n"], r["force"]): r for r in cells}
    for kernel in dict.fromkeys(r["kernel"] for r in cells + list(timed.values())):
        ns = sorted({k[1] for k in by if k[0] == kernel} | {k[1] for k in timed if k[0] == kernel})
        print(f"\n**{kernel}** (halo2 cells / Noir gates; auto = the compiler's choice; proofs of auto on halo2-base)\n")
        print("| n | auto halo2 cells (k) | " + " | ".join(forces) + " | auto Noir gates | " + " | ".join(forces)
              + " | auto prove s (create_proof / wall) | checks |")
        print("|---" * (5 + 2 * len(forces)) + "|")
        for n in ns:
            au = by.get((kernel, n, "auto")) or timed.get((kernel, n), {})
            t = timed.get((kernel, n), {})
            hc = [f(by.get((kernel, n, x), {}).get("cells")) for x in forces]
            ng = [f(by.get((kernel, n, x), {}).get("gates")) for x in forces]
            print(f"| {n} | **{f(au.get('cells'))}** (k {au.get('k')}) | " + " | ".join(hc)
                  + f" | **{f(au.get('gates'))}** | " + " | ".join(ng)
                  + f" | {f(t.get('create_proof_median_s'))} / {f(t.get('prove_median_s'))} | {ok(t)} |")


def compare(d, ref):
    diffs = []
    for fam in ("argsort", "matmul", "convolve"):
        a = {(r["n"], r["label"]): (r.get("cells"), r.get("k")) for r in rows(d / "challenge" / f"scale_{fam}.jsonl")}
        b = {(r["n"], r["label"]): (r.get("cells"), r.get("k")) for r in rows(ref / "challenge" / f"scale_{fam}.jsonl")}
        diffs += [(fam, k, a.get(k), b.get(k)) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
        diffs += [(fam, k, "missing") for k in sorted(set(b) - set(a))]
    a = {(r["kernel"], r["n"], r["force"]): (r.get("cells"), r.get("k"), r.get("gates")) for r in rows(d / "memory_cells.jsonl")}
    b = {(r["kernel"], r["n"], r["force"]): (r.get("cells"), r.get("k"), r.get("gates")) for r in rows(ref / "memory_cells.jsonl")}
    diffs += [("memory", k, a.get(k), b.get(k)) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    diffs += [("memory", k, "missing") for k in sorted(set(b) - set(a))]
    for name in ("scale_zinnia", "scale_hw"):
        a = {r["case"]: (r.get("cells"), r.get("k")) for r in rows(d / "hw" / f"{name}.jsonl")}
        b = {r["case"]: (r.get("cells"), r.get("k")) for r in rows(ref / "hw" / f"{name}.jsonl")}
        diffs += [(name, k, a.get(k), b.get(k)) for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    print(f"\nCell/gate counts differing from {ref}: {len(diffs)}")
    for x in diffs:
        print("  ", x)


d = Path(sys.argv[1])
challenge(d)
hw(d)
memory(d)
if "--ref" in sys.argv:
    compare(d, Path(sys.argv[sys.argv.index("--ref") + 1]))
