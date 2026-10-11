#!/usr/bin/env python3
"""Array-lowering ablation tables from a run_configs.sh output directory.

Usage: ablation_tables.py DIR   (DIR has costs_/gates_/measure_<config>.json and noir_hw.json)
"""
import json, math, sys
from pathlib import Path

D = Path(sys.argv[1])
CFGS = ['final', 'abl_joint', 'fix_delegate', 'fix_selects']
def gm(v): return math.exp(sum(map(math.log, v)) / len(v)) if v else float('nan')
def cells(r): return r['advice'] + r['lookup']
def j(name): return json.loads((D / name).read_text())
hw = j('noir_hw.json')
C = {c: j(f'costs_{c}.json')['cases'] for c in CFGS}
G = {c: j(f'gates_{c}.json') for c in CFGS}
M = {c: j(f'measure_{c}.json') for c in CFGS}
print('DS-1000 perf (100). halo2 cells at common width vs halo2_hw; Noir gates vs noir_hw')
print('config | halo2 gm vs hw | above hw | changed vs final | gm vs final | max vs final || noir gm vs hw | above hw | changed | gm vs final | max')
for c in CFGS:
    hr = [cells(r['common']['zinnia']) / cells(r['common']['halo2_hw']) for r in C[c].values()]
    fm, cm = M['final']['ds1000'], M[c]['ds1000']
    rel = [cm[k]['halo2']['cells'] / fm[k]['halo2']['cells'] for k in fm if cm.get(k, {}).get('halo2', {}).get('cells')]
    nr = [G[c][k]['gates'] / hw[k]['gates'] for k in hw if k in G[c] and k.startswith('ds1000')]
    nrel = [cm[k]['noir']['gates'] / fm[k]['noir']['gates'] for k in fm if cm.get(k, {}).get('noir', {}).get('gates')]
    print(f"{c} | {gm(hr):.4f} | {sum(r > 1 for r in hr)} | {sum(r != 1 for r in rel)} | {gm(rel):.4f} | {max(rel):.2f} || "
          f"{gm(nr):.4f} | {sum(r > 1 for r in nr)} | {sum(r != 1 for r in nrel)} | {gm(nrel):.4f} | {max(nrel):.2f}")
print('\nNPBench/Pythran perf (79), vs final (own-k cells / gates)')
for c in CFGS[1:]:
    fm, cm = M['final']['npbench_pythran'], M[c]['npbench_pythran']
    for t, key in [('halo2', 'cells'), ('noir', 'gates')]:
        rel = {k: cm[k][t][key] / fm[k][t][key] for k in fm
               if cm.get(k, {}).get(t, {}).get(key) and fm[k].get(t, {}).get(key)}
        ch = {k: v for k, v in rel.items() if v != 1}
        top = sorted(ch.items(), key=lambda x: -x[1])[:4]
        print(f"{c} {t}: n={len(rel)} changed {len(ch)} gm {gm(list(rel.values())):.4f} top {[(k, round(v, 2)) for k, v in top]}")
print('\nDS-1000 largest changes vs final')
for c in CFGS[1:]:
    fm, cm = M['final']['ds1000'], M[c]['ds1000']
    for t, key in [('halo2', 'cells'), ('noir', 'gates')]:
        rel = {k[-4:]: round(cm[k][t][key] / fm[k][t][key], 2) for k in fm if cm.get(k, {}).get(t, {}).get(key)}
        print(c, t, sorted([x for x in rel.items() if x[1] != 1], key=lambda x: -x[1])[:8])
