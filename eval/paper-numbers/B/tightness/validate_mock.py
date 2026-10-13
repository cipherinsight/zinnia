"""Check the worst-case inputs against the compiled circuits, not only NumPy.

For every program, every construction of constructions.py and every distinct input that
`run_search.py` reports as an array's maximum or as the simultaneous maximum is evaluated with the mock backend (constraint
satisfaction only: no keys, no proof) on
  * the observational probe circuit of ../final/probes/<program>.json (the circuit whose
    output layout gave the capacities), reading each array's runtime length from its
    exported length slot, and
  * for the three branch-merged crop results, the original program's circuit.
The input must satisfy the circuit, and every length must equal the NumPy size.
periodic_dist's probe evaluates the three periodic branches unconditionally, so its inputs
are checked with all three flags set (a flag gates only its own axis's arrays).

Usage: validate_mock.py <program id> | validate_mock.py   (writes mock/<program id>.json)
"""
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import common

OUT = common.HERE / 'mock'


def inputs_of(pid):
    """Distinct inputs to check: every construction (method (a)), and every input the
    search reports as an array's or the simultaneous maximum, with the reason."""
    from constructions import CONSTRUCTIONS
    P = common.Program(pid)
    found = {}
    for label, over, _ in CONSTRUCTIONS.get(P.entry.program, []):
        key = json.dumps({**P.fixture_input('A'), **over}, sort_keys=True)
        found.setdefault(key, set()).add(f'construction {label}')
    for path in sorted((common.HERE / 'search').glob(pid + '*.json')):
        r = json.loads(path.read_text())
        tag = path.name[len(pid):-5]
        for a in r['arrays']:
            key = json.dumps(a['best_input'], sort_keys=True)
            found.setdefault(key, set()).add(f'{a["array"]}{tag}')
        key = json.dumps(r['simultaneous']['input'], sort_keys=True)
        found.setdefault(key, set()).add(f'simultaneous{tag}')
    return [(json.loads(k), sorted(v)) for k, v in found.items()]


def one(pid):
    from benchmarking.harness.costs_halo2 import mock_publics_in_order
    from benchmarking.harness.kernels import compile_kernel, load_kernel_module
    from zinnia import ZKCircuit
    P = common.Program(pid)
    r = json.loads((common.FINAL / 'probes' / (pid + '.json')).read_text())
    base = ZKCircuit.from_method(getattr(load_kernel_module(P.entry.kernel_path), P.entry.function))
    probe_c = ZKCircuit(P.entry.function, r['source'], chips=list(base.chips.values()),
                        externals=list(base.externals.values())).compile()
    layout = {tuple(x['output'].get('return', [])): x for x in probe_c.get_output_layout()}
    crop_rows = [row for row in P.rows if row['source'] == 'original_output']
    orig_c = compile_kernel(P.entry.kernel_path, P.entry.function) if crop_rows else None
    # The capacities of this build must be the inventory's (measured on 2569c9ed).
    capacities = {}
    for row in P.rows:
        x = (next(x for x in orig_c.get_output_layout() if 'dynamic' in x) if row['source'] == 'original_output'
             else layout[tuple(row['path'])])
        capacities[row['array']] = len(x['dynamic']['cells']) // (2 if x.get('complex') else 1)
    capacity_mismatch = {a: c for a, c in capacities.items() if c != common.capacity(next(r for r in P.rows if r['array'] == a))}
    checks = []
    for values, purpose in inputs_of(pid):
        if P.entry.program == 'pythran_periodic_dist':
            values = {**values, 'periodicX': True, 'periodicY': True, 'periodicZ': True}
        numpy_sizes = P.sizes(values)
        args = P.arguments(values)
        lengths, error = {}, None
        try:
            mock = probe_c.prove(*args, backend='mock')
            ok = mock.proof_bytes_hex == 'mock_satisfied'
        except Exception as e:
            ok, error = False, str(e)[:500]
        if ok:
            publics = mock_publics_in_order(mock.public_values)
            for row in P.rows:
                if row['source'] != 'expression_probe':
                    continue
                d = layout[tuple(row['path'])]['dynamic']
                slot = d['length']
                lengths[row['array']] = slot['constant'] if 'constant' in slot else publics[slot['public']]
            if orig_c is not None:
                try:
                    m2 = orig_c.prove(*args, backend='mock')
                    ok = m2.proof_bytes_hex == 'mock_satisfied'
                except Exception as e:
                    ok, error = False, str(e)[:500]
                pub2 = mock_publics_in_order(m2.public_values) if ok else []
                part = next(x for x in orig_c.get_output_layout() if 'dynamic' in x)
                slot = part['dynamic']['length']
                for row in crop_rows if ok else []:
                    lengths[row['array']] = slot['constant'] if 'constant' in slot else pub2[slot['public']]
        match = ok and numpy_sizes is not None and all(lengths[a] == numpy_sizes[a] for a in lengths)
        checks.append(dict(purpose=purpose, satisfied=ok, error=error, lengths_match_numpy=match,
                           circuit_lengths=lengths, numpy_sizes=numpy_sizes))
    return dict(program=pid, inputs=len(checks), capacity_mismatch=capacity_mismatch,
                all_ok=not capacity_mismatch and all(c['satisfied'] and c['lengths_match_numpy'] for c in checks),
                checks=checks)


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        pid = sys.argv[1]
        try:
            r = one(pid)
        except BaseException as e:
            r = dict(program=pid, error=str(e), traceback=traceback.format_exc())
        (OUT / (pid + '.json')).write_text(json.dumps(r, indent=1))
        return

    def run(pid):
        cmd = ['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=16G',
               sys.executable, '-B', __file__, pid]
        with (OUT / (pid + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=log, env=os.environ)
        r = json.loads((OUT / (pid + '.json')).read_text())
        print(pid, r.get('inputs'), 'ok' if r.get('all_ok') else 'FAIL ' + str(r.get('error', '')), flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, common.PROGRAM_IDS))


if __name__ == '__main__':
    main()
