"""Row-domain and exact-Q32 regressions; only MockProver and nargo execute.

Run from the checkout with PYTHONPATH=.:benchmarking/baselines under memrun.sh:
  .venv/bin/python -m hw_check.rowdiv_fairness --work DIR --out results.json
The compiled Halo2 domain is audited from its constraints; the compiled Noir
circuit is also executed here because its unsigned radicand bound differs.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tomllib
from pathlib import Path

from benchmarking.harness.fixtures import load_fixture
from benchmarking.harness.kernels import compile_kernel
from .check import HALO2_BIN
from .entries import resolve
from .noir_driver import NARGO, compile_package, stage_package
from .prover_toml import SCALE, render_prover_toml
from .reduction_arithmetic import flatten


def trials(case):
    if case in ('0436', '0437'):
        return [
            ('ties', [2., 2., -3., -3., 0.], True, True),
            ('all_equal', [-1.] * 5, True, True),
            ('window_edges', [-2.**31, 2.**31 - 1, 0., 1., -1.], True, True),
            ('outside_window', [2.**31, 0., 1., 2., 3.], False, False),
            ('quantized_ties', [1., 1. + 2.**-34, -1., -1. - 2.**-34, 0.], True, True),
        ]
    if case == '0452':
        return [
            ('negative_floor', [-1, 2, 0, 0], True, True),
            ('zero', [0] * 4, False, False),
            ('int64_min', [-2**63, 0, 0, 0], False, False),
            ('l1_last', [2**63 - 2, -1, 0, 0], True, True),
            ('l1_overflow', [2**62, -2**62, 0, 0], False, False),
        ]
    return [
        ('negative_irrational', [-1, 1, 1, 0], True, True),
        ('perfect_square', [-3, 4, 0, 0], True, True),
        ('zero', [0] * 4, False, False),
        ('below_31', [32767, 32768, 0, 0], True, True),
        ('at_31', [32768, 32768, 0, 0], False, True),
        ('above_31', [32768, 32768, 1, 0], False, True),
        ('below_32', [65535, 0, 0, 0], False, True),
        ('at_32', [65536, 0, 0, 0], False, False),
        ('above_32', [65536, 1, 0, 0], False, False),
    ]


def expected(case, row):
    if case in ('0436', '0437'):
        from .prover_toml import quantize
        codes = [quantize(x) for x in row]
        extreme = (max if case == '0436' else min)(codes)
        return [int(x == extreme) for x in codes] * 10
    if case == '0452':
        return [x * SCALE // sum(abs(v) for v in row) for x in row] * 5
    root = math.isqrt(sum(x*x for x in row) * SCALE * SCALE)
    return [x * SCALE * SCALE // root for x in row] * 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    results = []
    for case in ('0436', '0437', '0452', '0453'):
        entry = resolve(case)
        fixture = json.loads(entry.fixture.read_text())
        compiled = compile_kernel(entry.kernel, 'kernel', backend='noir')
        source = compiled.emit_structural(*load_fixture(entry.fixture).arguments('B'), backend='noir')
        packages = {}
        baseline = stage_package(entry.directory, args.work / case / 'baseline', entry.function)
        packages['baseline_noir'] = (baseline, compile_package(baseline))
        pkg = args.work / case / 'compiled'
        (pkg / 'src').mkdir(parents=True, exist_ok=True)
        (pkg / 'src/main.nr').write_text(source)
        library = Path('benchmarking/noir_fpc').resolve()
        (pkg / 'Nargo.toml').write_text('[package]\nname = "circuit"\ntype="bin"\nauthors=[""]\n'
                                      f'[dependencies]\nnoir_fpc={{path="{library}"}}\n')
        packages['compiled_noir'] = (pkg, compile_package(pkg))
        for name, row, halo_ok, noir_ok in trials(case):
            param = 'a' if case in ('0436', '0437') else 'X'
            fixture['inputs']['B'] = {param: [row] * (10 if param == 'a' else 5)}
            fp = args.work / case / 'fixture.json'
            fp.write_text(json.dumps(fixture))
            proc = subprocess.run([str(HALO2_BIN), 'mock', *entry.program_args(),
                                   '--fixture', str(fp), '--input', 'B'], capture_output=True, text=True)
            (args.work / case / (name + '-halo2.log')).write_text(proc.stdout + proc.stderr)
            data = json.loads(next(line for line in reversed(proc.stdout.splitlines()) if line.startswith('{')))
            assert 'satisfied' in data, (case, name, data)
            accepted = bool(data['satisfied'])
            assert accepted == halo_ok, (case, name, 'halo2', data)
            if accepted:
                want = expected(case, row)
                actual = data['publics']
                if param == 'X':
                    actual = [round(x * SCALE) for x in actual]
                assert actual == want, (case, name, actual, want)
            result = dict(case=case, trial=name, baseline_halo2=accepted)
            for system, (pkg, artifact) in packages.items():
                (pkg / 'Prover.toml').write_text(render_prover_toml(artifact['abi'], fixture, 'B'))
                proc = subprocess.run([NARGO, 'execute', '--silence-warnings', '--overwrite-return'],
                                      cwd=pkg, capture_output=True, text=True)
                (args.work / case / (name + '-' + system + '.log')).write_text(proc.stdout + proc.stderr)
                accepted = proc.returncode == 0
                assert accepted == noir_ok, (case, name, system, proc.stdout, proc.stderr)
                if accepted:
                    actual = flatten(tomllib.loads((pkg / 'Prover.toml').read_text())['return'])
                    assert actual == expected(case, row), (case, name, system, actual)
                result[system] = accepted
            results.append(result)
            print(json.dumps(result), flush=True)
            args.out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
