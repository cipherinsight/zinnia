"""Exploratory paired analysis of an immutable study backup; no live DB writes.
Run with system Python (requires NumPy): python analyze.py
"""
import csv
import hashlib
import itertools
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhantong/Projects/zinnia-project')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'study-backups/all-results-20261006-173802/study.sqlite3'
PRE_REDO = ROOT / 'study-backups/before-rust-redo-P1-P6-20261006-164254/study.sqlite3'
METRICS = ['active_seconds', 'first_correct_capped_seconds', 'difficulty', 'attempts', 'success']


def extract(path):
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    data = []
    backgrounds = {}
    for p in db.execute('SELECT * FROM participants WHERE cohort_id=?', ('crossover-01e0fda0f350',)):
        backgrounds[p['label']] = json.loads(p['background'] or '{}')
        for s in db.execute('SELECT * FROM sessions WHERE token=? ORDER BY period', (p['token'],)):
            if s['task'] not in json.loads(p['task_order']):
                continue
            assert s['finished'] is not None
            runs = [dict(r) for r in db.execute('SELECT * FROM runs WHERE token=? AND task=? ORDER BY requested', (p['token'], s['task']))]
            for r in runs:
                r['result'] = json.loads(r['result'] or '{}')
            attempts = [r for r in runs if r['purpose'] in ('examples', 'custom')]
            grades = [r for r in runs if r['purpose'] == 'grade']
            assert len(grades) == 1
            hidden = {r['parent_run_id']: r for r in runs if r['purpose'] == 'attempt_grade'}
            # Require both visible and hidden success; hidden diffusion tests alone
            # are known to miss some incorrect in-place implementations.
            first = [r['active_elapsed'] for r in attempts
                     if r['result'].get('status') == 'passed'
                     and hidden.get(r['id'], {}).get('result', {}).get('status') == 'passed'
                     and r['active_elapsed'] is not None]
            final = grades[-1]['result'].get('status') == 'passed'
            elapsed = s['finished'] - s['started'] - s['paused_total']
            if final:
                first.append(elapsed)
            response = json.loads(s['response'] or '{}')
            data.append(dict(label=p['label'], task=s['task'], language=s['condition'],
                             period=s['period'], active_seconds=min(600., max(0., elapsed)),
                             first_correct_capped_seconds=min([600.] + first),
                             ever_correct=int(bool(first)), outcome=s['outcome'],
                             difficulty=response.get('difficulty'), attempts=len(attempts),
                             compile_failures=sum(r['result'].get('status') == 'compile_error' for r in attempts),
                             success=int(final), final_status=grades[-1]['result'].get('status')))
    db.close()
    assert len(data) == 72
    return data, backgrounds


def paired(data, metric, cluster_same_person=False):
    pairs = []
    for label in sorted({r['label'] for r in data}, key=lambda s: int(s[1:])):
        rows = [r for r in data if r['label'] == label]
        # Difficulty uses only schedules with ALL six ratings, avoiding
        # treating unobserved timeout difficulty as a neutral/easy rating.
        if any(r[metric] is None for r in rows):
            continue
        assert len(rows) == 6
        vals = [np.mean([r[metric] for r in rows if r['language'] == lang]) for lang in ('zinnia', 'rust')]
        pairs.append((label, *vals))
    if cluster_same_person:
        same = [p for p in pairs if p[0] in ('P3', 'P11')]
        pairs = [p for p in pairs if p[0] not in ('P3', 'P11')]
        if same:
            pairs.append(('P3+P11', *np.mean([p[1:] for p in same], axis=0)))
    return pairs


def stats(pairs, seed=20261006):
    z = np.array([p[1] for p in pairs]); r = np.array([p[2] for p in pairs]); d = r-z
    n = len(d)
    if not n:
        return None
    # Exact enumeration for the sign-flip test under a symmetric paired-null
    # model. This is NOT a randomization test of the actual schedule assignment.
    signs = np.array(list(itertools.product((-1, 1), repeat=n)))
    null = signs @ d / n
    observed = float(d.mean())
    two = float(np.mean(abs(null) >= abs(observed)-1e-10))
    one = float(np.mean(null >= observed-1e-10))
    nz = d[abs(d) > 1e-10]; k = sum(nz > 0); m = len(nz)
    sign_p = min(1., 2*sum(math.comb(m, j) for j in range(min(k, m-k)+1))/2**m) if m else 1.
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(100000, n))].mean(axis=1)
    return dict(n=n, zinnia_mean=float(z.mean()), rust_mean=float(r.mean()),
                rust_minus_zinnia=observed, bootstrap_95ci=np.quantile(boot, [.025, .975]).tolist(),
                signflip_p_two_sided=two, signflip_p_one_sided=one, sign_test_p_two_sided=sign_p,
                positive=int(sum(d>1e-10)), negative=int(sum(d < -1e-10)), ties=int(sum(abs(d)<=1e-10)),
                participant_differences={p[0]: float(di) for p, di in zip(pairs, d)})


def adjusted_time(data):
    # Descriptive task/participant/period fixed-effects adjustment; no
    # unclustered row-level p-values are used with these repeated measures.
    labels = sorted({r['label'] for r in data}); tasks = sorted({r['task'] for r in data})
    x = np.array([[1., r['language']=='rust'] + [r['label']==p for p in labels[1:]]
                  + [r['task']==t for t in tasks[1:]] + [r['period']==j for j in range(2, 7)] for r in data], float)
    y = np.array([r['active_seconds'] for r in data])
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def analyze(data, cluster=False):
    result = {m: stats(paired(data, m, cluster)) for m in METRICS}
    # Conservative familywise correction across all five reported endpoints.
    ordered = sorted(METRICS, key=lambda m: result[m]['signflip_p_two_sided'])
    floor = 0.
    for j, m in enumerate(ordered):
        floor = max(floor, min(1., (len(ordered)-j)*result[m]['signflip_p_two_sided']))
        result[m]['holm_p_two_sided'] = floor
    result['descriptive_adjusted_rust_time_excess_seconds'] = adjusted_time(data)
    return result


data, backgrounds = extract(SOURCE)
pre, _ = extract(PRE_REDO)
replacements = {('P1', 'image_filter_v5'), ('P6', 'block_energy_v5')}
restored = [r for r in data if (r['label'], r['task']) not in replacements] + [r for r in pre if (r['label'], r['task']) in replacements]
scenarios = {
    'latest_all_12': (data, False),
    'restore_two_original_rust_timeouts_all_12': (restored, False),
    'exclude_known_reset_schedules_sensitivity': ([r for r in data if r['label'] not in {'P1','P2','P4','P6'}], False),
}
results = {name: analyze(rows, cluster) for name, (rows, cluster) in scenarios.items()}
with (OUT/'task_data.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'results.json').write_text(json.dumps(dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                                              cohort_note='All 12 are distinct valid study participants, including P0. Zhaoyu distributed P3/P11 to different people.',
                                              backgrounds=backgrounds, analyses=results), indent=2)+'\n')
for name, result in results.items():
    print('\n'+name)
    for metric in METRICS:
        s=result[metric]
        print(metric, 'n',s['n'], 'Z/R',round(s['zinnia_mean'],3),round(s['rust_mean'],3),
              'diff',round(s['rust_minus_zinnia'],3),'CI',[round(v,3) for v in s['bootstrap_95ci']],
              'p',s['signflip_p_two_sided'],'Holm',s['holm_p_two_sided'],'sign',s['sign_test_p_two_sided'])
    print('adjusted time difference',result['descriptive_adjusted_rust_time_excess_seconds'])
