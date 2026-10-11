"""Ablation table over the perf set from one directory of perf_configs.sh outputs, and a
case-by-case comparison with the local measurements.

Usage: python3 perf_compare.py REMOTE_DIR [LOCAL_DIR]
  REMOTE_DIR: costs_<c>.json, gates_<c>.json, ir_<c>.json, noir_hw.json (c = final, abl_smt,
              abl_layer, abl_check, abl_af)
  LOCAL_DIR:  eval-redo/fix/fact-review-data (costs_<c>.json, gates_<c>.json measured at
              dad059e2, IR identical; ir_f8.json / ir_abl_<x>_f8.json at 35c86a81)
Ratios: halo2-base Zinnia / halo2_hw advice+lookup cells at the common lookup width; Noir
Zinnia / noir_hw bb gates. Checks = Assert* IR statements summed over the 100 cases.
"""
import json
import math
import sys
from pathlib import Path

CONFIGS = ("final", "abl_smt", "abl_layer", "abl_check", "abl_af")


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def jload(p):
    return json.load(open(p)) if Path(p).exists() else None


def cells_common(r):
    c = r.get("common") or {}
    if not c.get("zinnia") or not c.get("halo2_hw"):
        return None, None
    return (c["zinnia"]["advice"] + c["zinnia"]["lookup"], c["halo2_hw"]["advice"] + c["halo2_hw"]["lookup"])


def cells_own(r):
    z = r.get("zinnia") or {}
    return z["advice"] + z["lookup"] if "advice" in z else None


def summary(d, c, hw_gates):
    costs, gates, ir = jload(d / f"costs_{c}.json"), jload(d / f"gates_{c}.json"), jload(d / f"ir_{c}.json")
    row = {"config": c}
    if costs:
        perf = {k: r for k, r in costs["cases"].items() if r.get("status") == "perf"}
        ratios = {}
        for k, r in perf.items():
            z, h = cells_common(r)
            if z and h:
                ratios[k] = z / h
        row.update(halo2_measured=len(ratios), halo2_geomean=round(geomean(ratios.values()), 4),
                   halo2_above=sorted(f"{k[-4:]}:{v:.3f}" for k, v in ratios.items() if v > 1),
                   halo2_failed=sorted(k[-4:] for k in perf if k not in ratios))
    if gates:
        ratios = {k: g["gates"] / hw_gates[k] for k, g in gates.items()
                  if isinstance(g, dict) and g.get("gates") and k in hw_gates}
        row.update(noir_measured=len(ratios), noir_geomean=round(geomean(ratios.values()), 4),
                   noir_above=sorted(f"{k[-4:]}:{v:.4f}" for k, v in ratios.items() if v > 1),
                   noir_failed=sorted(k[-4:] for k, g in gates.items() if not (isinstance(g, dict) and g.get("gates"))))
    if ir:
        for t in ("halo2", "noir"):
            rows = [v[t] for v in ir.values() if "asserts" in v[t]]
            row[f"checks_{t}"] = sum(v["asserts"] for v in rows)
            row[f"checks_nondomain_{t}"] = sum(v["asserts"] - v["domain_asserts"] for v in rows)
            row[f"compile_errors_{t}"] = sorted(k[-4:] for k, v in ir.items() if "error" in v[t])
    return row


def local_ir_name(c):
    return "ir_f8.json" if c == "final" else f"ir_{c}_f8.json"


def compare(remote, local, c):
    out = []
    ri, li = jload(remote / f"ir_{c}.json"), jload(local / local_ir_name(c))
    if ri and li:
        diff = [(k, t) for k in sorted(set(ri) | set(li)) for t in ("halo2", "noir")
                if (ri.get(k, {}).get(t, {}).get("sha") or ri.get(k, {}).get(t, {}).get("error", "")[:60])
                != (li.get(k, {}).get(t, {}).get("sha") or li.get(k, {}).get(t, {}).get("error", "")[:60])]
        out.append(f"IR digest differs from local ({local_ir_name(c)}) on {len(diff)} of {2 * len(li)} (case, target): {diff[:20]}")
    rc, lc = jload(remote / f"costs_{c}.json"), jload(local / f"costs_{c}.json")
    if rc and lc:
        diff = []
        for k, r in rc["cases"].items():
            l = lc["cases"].get(k)
            if l is None:
                diff.append((k, "missing locally"))
                continue
            a, b = (cells_own(r), cells_common(r)), (cells_own(l), cells_common(l))
            if a != b:
                diff.append((k, a, b))
        out.append(f"halo2 cells (own k; common width Zinnia, halo2_hw) differ from local on {len(diff)} cases: {diff[:20]}")
    rg, lg = jload(remote / f"gates_{c}.json"), jload(local / f"gates_{c}.json")
    if rg and lg:
        diff = [(k, (rg.get(k) or {}).get("gates"), (lg.get(k) or {}).get("gates")) for k in sorted(set(rg) | set(lg))
                if (rg.get(k) or {}).get("gates") != (lg.get(k) or {}).get("gates")]
        out.append(f"Noir gates differ from local on {len(diff)} cases: {diff[:20]}")
    return out


def main():
    remote = Path(sys.argv[1])
    local = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    hw = jload(remote / "noir_hw.json") or (jload(local / "noir_hw.json") if local else {})
    hw_gates = {c: r["gates"] for c, r in hw.items() if isinstance(r, dict) and "gates" in r}
    if local and jload(remote / "noir_hw.json") and jload(local / "noir_hw.json"):
        lh = jload(local / "noir_hw.json")
        d = [k for k in sorted(set(hw) | set(lh)) if (hw.get(k) or {}).get("gates") != (lh.get(k) or {}).get("gates")]
        print(f"noir_hw gates differ from local on {len(d)} cases: {d[:20]}")
    for c in CONFIGS:
        print(json.dumps(summary(remote, c, hw_gates)))
        if local:
            for line in compare(remote, local, c):
                print("  ", line)


main()
