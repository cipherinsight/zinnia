"""Aggregate search/, smt.json and mock/ into tightness.csv, summary.json and the generated
tables of REPORT.md (between the `<!-- generated:begin -->` / `<!-- generated:end -->`
markers). Prints the summary.
"""
import csv
import json
import math
import statistics

import common
from constructions import CONSTRUCTIONS

H = common.HERE
ALT = {'pythran__pythran_periodic_dist__dist': 'L>0'}


def load(pid, alt=None):
    suffix = '.' + alt if alt else ''
    return json.loads((H / 'search' / (pid + suffix + '.json')).read_text())


def program_numbers(r):
    arrs = r['arrays']
    cap = sum(a['capacity'] for a in arrs)
    return dict(
        arrays=len(arrs),
        capacity=cap,
        # The paper's fill at the test inputs: elements over capacity, summed over the
        # program's arrays, mean of A and B.
        test_fill=sum(a['length_A'] + a['length_B'] for a in arrs) / (2 * cap),
        test_fill_per_array=statistics.mean((a['length_A'] + a['length_B']) / (2 * a['capacity']) for a in arrs),
        # Each array at its own worst-case input: mean of the per-array fractions ...
        worst_per_array=statistics.mean(a['max_size'] / a['capacity'] for a in arrs),
        # ... and summed like the paper's fill.
        worst_summed=sum(a['max_size'] for a in arrs) / cap,
        # One input for the whole program, maximising the summed fill.
        worst_simultaneous=r['simultaneous']['total'] / cap,
        all_reach_capacity=all(a['max_size'] == a['capacity'] for a in arrs),
        arrays_at_capacity=sum(a['max_size'] == a['capacity'] for a in arrs),
        simultaneous_full=r['simultaneous']['total'] == cap,
    )


def ratio(cap, n):
    return math.inf if n == 0 else cap / n


def aggregates(per):
    progs = list(per.values())
    arrays = [a for r in RESULTS.values() for a in r['arrays']]

    def mean(k):
        return statistics.mean(p[k] for p in progs)
    # Capacity / elements per program, as the paper's median 2.0 (geometric mean over the
    # program's arrays), here at each array's worst-case input.
    worst_ratio = []
    for pid, r in RESULTS.items():
        v = [ratio(a['capacity'], a['max_size']) for a in r['arrays']]
        worst_ratio.append(math.inf if math.inf in v else common.gmean(v))
    return dict(
        programs=len(progs), arrays=sum(p['arrays'] for p in progs),
        test_fill_mean=mean('test_fill'),
        test_fill_per_array_mean=mean('test_fill_per_array'),
        worst_per_array_mean=mean('worst_per_array'),
        worst_summed_mean=mean('worst_summed'),
        worst_simultaneous_mean=mean('worst_simultaneous'),
        worst_simultaneous_median=statistics.median(p['worst_simultaneous'] for p in progs),
        arrays_at_capacity=sum(p['arrays_at_capacity'] for p in progs),
        arrays_fraction_mean=statistics.mean(a['max_size'] / a['capacity'] for a in arrays),
        programs_all_arrays_at_capacity=sum(p['all_reach_capacity'] for p in progs),
        programs_simultaneously_full=sum(p['simultaneous_full'] for p in progs),
        median_program_capacity_over_elements_worst=statistics.median(worst_ratio),
        max_program_capacity_over_elements_worst=max(worst_ratio),
    )


def construction_label(pid, a):
    name = common.PROGRAMS[pid].program
    if a['construct'] is None or a['construct'] < a['max_size']:
        return None
    # The construction that reached the maximum (constructions are measured in order).
    P = PROGRAM_OBJ[pid]
    for label, over, _ in CONSTRUCTIONS.get(name, []):
        s = P.sizes({**P.fixture_input('A'), **over})
        if s and s[a['array']] == a['max_size']:
            return label
    return None


def fmt(x):
    return f'{100 * x:.1f}%'


RESULTS = {pid: load(pid) for pid in common.PROGRAM_IDS}
PROGRAM_OBJ = {pid: common.Program(pid) for pid in common.PROGRAM_IDS}


def main():
    per = {pid: program_numbers(r) for pid, r in RESULTS.items()}
    agg = aggregates(per)
    # Alternative domain: periodic_dist with L > 0.
    saved = dict(RESULTS)
    for pid, alt in ALT.items():
        RESULTS[pid] = load(pid, alt)
    per_alt = {pid: program_numbers(r) for pid, r in RESULTS.items()}
    agg_alt = aggregates(per_alt)
    RESULTS.clear()
    RESULTS.update(saved)

    mock = {pid: json.loads((H / 'mock' / (pid + '.json')).read_text()) for pid in common.PROGRAM_IDS}
    smt = json.loads((H / 'smt.json').read_text()) if (H / 'smt.json').exists() else {}

    alt_size = {(pid, b['array']): b['max_size'] for pid, alt in ALT.items() for b in load(pid, alt)['arrays']}
    rows = []
    for pid, r in RESULTS.items():
        for a in r['arrays']:
            rows.append(dict(program=pid, array=a['array'], expression=a['expression'], capacity=a['capacity'],
                             length_A=a['length_A'], length_B=a['length_B'], max_size=a['max_size'],
                             max_over_capacity=round(a['max_size'] / a['capacity'], 6),
                             construction=construction_label(pid, a) or '',
                             construction_size=a['construct'], generic_search_size=a['generic'],
                             max_size_alt=alt_size.get((pid, a['array']))))
    with (H / 'tightness.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = dict(aggregates=agg, aggregates_periodic_L_positive=agg_alt, per_program=per,
                   per_program_periodic_L_positive={k: per_alt[k] for k in ALT},
                   mock_all_ok=all(m.get('all_ok') for m in mock.values()),
                   mock_inputs=sum(m.get('inputs', 0) for m in mock.values()),
                   smt={k: {x: v[x] for x in ('omt', 'optimum', 'checked_bound', 'above_bound', 'seconds', 'numpy')} for k, v in smt.items()})
    (H / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(aggregates=agg, alt=agg_alt, mock_all_ok=summary['mock_all_ok'],
                          mock_inputs=summary['mock_inputs']), indent=1))

    # ---- generated tables --------------------------------------------------------------
    out = ['### Per program', '',
           '| Program | Arrays | Capacity | Fill A/B (paper) | Worst, per-array mean | Worst, summed | Worst, one input | Arrays at capacity |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pid, p in per.items():
        out.append(f'| {common.short(pid)} | {p["arrays"]} | {p["capacity"]} | {fmt(p["test_fill"])} | '
                   f'{fmt(p["worst_per_array"])} | {fmt(p["worst_summed"])} | {fmt(p["worst_simultaneous"])} | '
                   f'{p["arrays_at_capacity"]}/{p["arrays"]} |')
    for pid, alt in ALT.items():
        p = per_alt[pid]
        out.append(f'| {common.short(pid)} ({alt}) | {p["arrays"]} | {p["capacity"]} | {fmt(p["test_fill"])} | '
                   f'{fmt(p["worst_per_array"])} | {fmt(p["worst_summed"])} | {fmt(p["worst_simultaneous"])} | '
                   f'{p["arrays_at_capacity"]}/{p["arrays"]} |')
    out += ['', '### Per array', '',
            'Max = largest size found over all methods; "construction" names the method-(a) input '
            'that reaches it (constructions.py); "search" is the largest size the program-agnostic '
            'search (sweep, random, hill climbing) found on its own.', '',
            '| Program | Array (source expression, occurrence) | Capacity | A | B | Max | Max/cap | Construction | Search |',
            '|---|---|---:|---:|---:|---:|---:|---|---:|']
    for x in rows:
        occ = ' #' + x['array'].split('#', 1)[1] if '#' in x['array'] else ''
        expr = x['expression'].replace('|', '\\|')
        alt = f' ({x["max_size_alt"]} with L>0)' if x['max_size_alt'] is not None and x['max_size_alt'] != x['max_size'] else ''
        out.append(f'| {common.short(x["program"])} | `{expr}`{occ} | {x["capacity"]} | {x["length_A"]} | '
                   f'{x["length_B"]} | {x["max_size"]}{alt} | {fmt(x["max_over_capacity"])} | {x["construction"]} | '
                   f'{x["generic_search_size"]} |')
    text = (H / 'REPORT.md').read_text() if (H / 'REPORT.md').exists() else '<!-- generated:begin -->\n<!-- generated:end -->\n'
    a, b = text.index('<!-- generated:begin -->'), text.index('<!-- generated:end -->')
    text = text[:a] + '<!-- generated:begin -->\n' + '\n'.join(out) + '\n' + text[b:]
    (H / 'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
