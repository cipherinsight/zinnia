"""Compare handwritten Noir vs Zinnia-emitted Noir on the DS-1000 sweep."""
import json, math, statistics
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent

hw = json.loads((HERE / "noir_baseline_results_fresh.json").read_text())["results"]
zn = json.loads((HERE / "zinnia_noir_results_final.json").read_text())["results"]

hw_by = {r["case"]: r for r in hw}
zn_by = {r["case"]: r for r in zn}

print("=" * 78)
print("STATUS")
print("=" * 78)
print(f"  handwritten: {Counter(r['status'] for r in hw)}")
print(f"  zinnia-noir: {Counter(r['status'] for r in zn)}")

shared = sorted(set(hw_by) & set(zn_by) & {c for c in hw_by if hw_by[c]['status']=='ok'} & {c for c in zn_by if zn_by[c]['status']=='ok'})
print(f"\n  cases ok on both: {len(shared)} (used for ratio statistics)")
only_zn = sorted(c for c in zn_by if c not in hw_by or hw_by[c]['status']!='ok')
only_hw = sorted(c for c in hw_by if c not in zn_by or zn_by[c]['status']!='ok')
print(f"  ok on Zinnia, not on handwritten: {only_zn}")
print(f"  ok on handwritten, not on Zinnia: {only_hw}")

# Build ratio table
rows = []
for c in shared:
    h = hw_by[c]
    z = zn_by[c]
    hp = h["prove_seconds_median"]
    zp = z["prove_s"]
    ratio = zp / hp if hp > 0 else float('nan')
    rows.append({
        "case": c,
        "hw_prove_ms": hp * 1000,
        "zn_prove_ms": zp * 1000,
        "ratio": ratio,
        "hw_gates": h.get("gate_count"),
        "hw_proof": h.get("proof_bytes"),
        "zn_proof": z.get("proof_bytes"),
    })

ratios = [r["ratio"] for r in rows if r["ratio"] > 0]
hw_proves = [r["hw_prove_ms"] for r in rows]
zn_proves = [r["zn_prove_ms"] for r in rows]

print()
print("=" * 78)
print("PROVE-TIME COMPARISON  (median over 3 reruns, ms)")
print("=" * 78)
print(f"  handwritten:   min={min(hw_proves):6.1f}  median={statistics.median(hw_proves):6.1f}  "
      f"p90={statistics.quantiles(hw_proves, n=10)[-1]:7.1f}  max={max(hw_proves):8.1f}  "
      f"sum={sum(hw_proves)/1000:7.2f}s")
print(f"  zinnia-emit:   min={min(zn_proves):6.1f}  median={statistics.median(zn_proves):6.1f}  "
      f"p90={statistics.quantiles(zn_proves, n=10)[-1]:7.1f}  max={max(zn_proves):8.1f}  "
      f"sum={sum(zn_proves)/1000:7.2f}s")

geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
arith = sum(ratios) / len(ratios)
print()
print(f"  ratio (Zinnia / handwritten):")
print(f"    geomean = {geomean:.3f}x      arith mean = {arith:.3f}x")
print(f"    median  = {statistics.median(ratios):.3f}x   min = {min(ratios):.3f}x   max = {max(ratios):.3f}x")

# Bucket cases
faster = [r for r in rows if r["ratio"] < 0.9]
parity = [r for r in rows if 0.9 <= r["ratio"] <= 1.1]
slower = [r for r in rows if r["ratio"] > 1.1]
print()
print(f"  Zinnia ≥10% faster ({len(faster)}):  ratio < 0.9")
print(f"  parity ±10%        ({len(parity)}):")
print(f"  Zinnia >10% slower ({len(slower)}): ratio > 1.1")

print()
print("=" * 78)
print("TOP 15 BIGGEST ABSOLUTE PROVE-TIME OUTLIERS (handwritten)")
print("=" * 78)
print(f"  {'case':<6} {'hw ms':>9} {'zn ms':>9} {'ratio':>7}  {'hw_gates':>10}")
for r in sorted(rows, key=lambda r: -r["hw_prove_ms"])[:15]:
    print(f"  {r['case']:<6} {r['hw_prove_ms']:>9.1f} {r['zn_prove_ms']:>9.1f} "
          f"{r['ratio']:>7.2f}x {r['hw_gates'] or 0:>10}")

print()
print("=" * 78)
print("ZINNIA HEADWINDS — biggest slowdowns vs handwritten")
print("=" * 78)
print(f"  {'case':<6} {'hw ms':>9} {'zn ms':>9} {'ratio':>7}  {'hw_gates':>10}")
for r in sorted(rows, key=lambda r: -r["ratio"])[:15]:
    print(f"  {r['case']:<6} {r['hw_prove_ms']:>9.1f} {r['zn_prove_ms']:>9.1f} "
          f"{r['ratio']:>7.2f}x {r['hw_gates'] or 0:>10}")

print()
print("=" * 78)
print("ZINNIA TAILWINDS — biggest wins vs handwritten")
print("=" * 78)
print(f"  {'case':<6} {'hw ms':>9} {'zn ms':>9} {'ratio':>7}  {'hw_gates':>10}")
for r in sorted(rows, key=lambda r: r["ratio"])[:15]:
    print(f"  {r['case']:<6} {r['hw_prove_ms']:>9.1f} {r['zn_prove_ms']:>9.1f} "
          f"{r['ratio']:>7.2f}x {r['hw_gates'] or 0:>10}")

print()
print("=" * 78)
print("PROOF SIZE")
print("=" * 78)
hw_proof_sizes = Counter(r["hw_proof"] for r in rows)
zn_proof_sizes = Counter(r["zn_proof"] for r in rows)
print(f"  handwritten proof_bytes distribution: {dict(hw_proof_sizes)}")
print(f"  zinnia-emit proof_bytes distribution: {dict(zn_proof_sizes)}")
mismatches = [(r["case"], r["hw_proof"], r["zn_proof"]) for r in rows if r["hw_proof"] != r["zn_proof"]]
print(f"  proof_bytes mismatches: {len(mismatches)}  (first 5: {mismatches[:5]})")

print()
print("=" * 78)
print("GATE COUNT (handwritten side only; emitter doesn't surface it)")
print("=" * 78)
hg = [r["hw_gates"] for r in rows if r["hw_gates"]]
print(f"  handwritten gates: min={min(hg)}  median={statistics.median(hg):.0f}  "
      f"max={max(hg)}  sum={sum(hg):,}")

print()
print("=" * 78)
print("KERNEL-FAMILY ROLL-UP  (geomean ratio per family)")
print("=" * 78)
families = {
    "passthrough/reshape":  ["0301","0302","0303","0304","0317","0349","0500","0385","0388","0389","0390","0391","0392","0393"],
    "argmin/argmax/unravel":["0310","0311","0312","0313","0314","0320","0321","0322"],
    "flatten/transpose":    ["0340","0341","0342","0343"],
    "argsort/sort+gather":  ["0355","0356","0357","0358","0381","0382","0383"],
    "selection-sort N=100": ["0381","0382","0383"],
    "transcendentals":      ["0324","0325","0326"],
    "isclose / corrcoef":   ["0368","0369","0370","0438","0491","0492"],
    "insert/delete":        ["0359","0360","0361","0363","0364","0365"],
    "conditional filter":   ["0412","0431","0442","0443","0444","0462","0463","0481"],
    "bin mean/max":         ["0414","0415","0417","0419"],
    "iir recursion":        ["0398","0399"],
    "static gather":        ["0450","0501","0502","0503","0504","0505"],
    "nonzero crop":         ["0507","0508","0509","0510"],
    "3x3 sliding":          ["0467","0468"],
}
for fam, cs in families.items():
    rs = [r["ratio"] for r in rows if r["case"] in cs]
    if not rs:
        continue
    gm = math.exp(sum(math.log(x) for x in rs) / len(rs))
    hw_sum = sum(r["hw_prove_ms"] for r in rows if r["case"] in cs)
    zn_sum = sum(r["zn_prove_ms"] for r in rows if r["case"] in cs)
    print(f"  {fam:<25}  n={len(rs):>2}  geomean={gm:.2f}x  hw_total={hw_sum:>7.0f}ms  zn_total={zn_sum:>7.0f}ms")
