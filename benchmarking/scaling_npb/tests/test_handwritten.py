"""Source safety and per-entry integration checks; no prover or native build."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarking.scaling_npb import handwritten as hw


@pytest.fixture
def sizes():
    # Small implementation of the external sizes.py interface. Full formula
    # language belongs to hw_check; these tests target the independent checker.
    pattern = re.compile(
        r"^(?P<head>(?:const|global) (?P<name>[A-Z_]+): \w+ = )"
        r"(?P<value>\d+)(?P<tail>; // = (?P<formula>.+))$")
    def constants(lines):
        return [SimpleNamespace(line=i + 1, name=m['name'], value=int(m['value']), formula=m['formula'])
                for i, line in enumerate(lines) if (m := pattern.match(line))]
    def dimensions(fixture):
        return {f"{p['name']}_{axis}": n for p in fixture['parameters']
                for axis, n in enumerate(p['shape'])}
    def evaluate(formula, dims):
        if formula == 'factorial(a_0)':
            import math
            return math.factorial(dims['a_0'])
        return dims[formula]
    def instantiate(text, fixture):
        return '\n'.join(pattern.sub(lambda m: m['head'] + str(evaluate(m['formula'], dimensions(fixture)))
                                     + m['tail'], line) for line in text.split('\n'))
    return SimpleNamespace(DECLARATION=pattern, size_constants=constants, dimensions=dimensions,
                           evaluate=evaluate, instantiate=instantiate, check=lambda *_: ([], []))


def fixture(n):
    return {'parameters': [{'name': 'a', 'kind': 'array', 'shape': [n]}]}


SOURCE = 'const N: usize = 4; // = a_0\nfn main() { for i in 0..N { f(i, 4); } }\n'


def test_shape_formula_and_factorial(sizes):
    source = SOURCE + 'global PERMS: u32 = 24; // = factorial(a_0)\n'
    candidate = sizes.instantiate(source, fixture(5))
    changes = hw.check_rewrite(source, candidate, fixture(4), fixture(5), sizes)
    assert [c['new'] for c in changes] == [5, 120]
    assert hw.check_rewrite(source, source, fixture(4), fixture(4), sizes)


@pytest.mark.parametrize('replace', [
    lambda s: s.replace('f(i, 4)', 'f(i, 5)'),
    lambda s: s.replace('f(i, 4)', 'g(i, 4)'),
    lambda s: s.replace('for i', 'for  i'),
    lambda s: s.replace('// = a_0', '// = factorial(a_0)'),
    lambda s: s + '// extra comment\n',
])
def test_reject_non_size_change(sizes, replace):
    candidate = replace(sizes.instantiate(SOURCE, fixture(8)))
    with pytest.raises(ValueError, match='non-size'):
        hw.check_rewrite(SOURCE, candidate, fixture(4), fixture(8), sizes)


def test_reject_wrong_values_and_overflow(sizes):
    with pytest.raises(ValueError, match='but a_0 = 4'):
        hw.check_rewrite(SOURCE.replace('= 4;', '= 3;'), sizes.instantiate(SOURCE, fixture(8)),
                         fixture(4), fixture(8), sizes)
    with pytest.raises(ValueError, match='does not fit u8'):
        source = SOURCE.replace('usize', 'u8')
        hw.check_rewrite(source, sizes.instantiate(source, fixture(256)), fixture(4), fixture(256), sizes)
    with pytest.raises(ValueError, match='no annotated'):
        hw.check_rewrite('fn main() {}', 'fn main() {}', fixture(4), fixture(8), sizes)


def test_entry_names_coverage_write_and_conflict(tmp_path, monkeypatch, sizes):
    root, data = tmp_path / 'root', tmp_path / 'data'
    bench = root / 'benchmarking'
    original = bench / 'datasets/pythran/demo'
    target = data / 'cap32/demo'
    original.mkdir(parents=True)
    target.mkdir(parents=True)
    (bench / 'datasets/case_sets').mkdir()
    raw = {'case': 'demo__first', 'path': 'datasets/pythran/demo', 'function': 'first',
           'fixture': 'fixture.first.json'}
    (bench / 'datasets/case_sets/npbench_pythran_perf.json').write_text(json.dumps({'cases': [raw]}))
    (original / 'fixture.first.json').write_text(json.dumps(fixture(4)))
    (target / 'fixture.first.json').write_text(json.dumps(fixture(8)))
    (original / 'target.first.rs').write_text(SOURCE)
    point = {'case': raw['case'] + '_cap32', 'cap': 32, 'status': 'scaled', 'folder': 'cap32/demo',
             'fixture': 'fixture.first.json'}
    (data / 'points.json').write_text(json.dumps({'points': [point, dict(point, cap=64, status='unchanged')]}))
    monkeypatch.setattr(hw, 'load_sizes', lambda _: sizes)
    report = hw.generate(data, root)
    assert report['counts'] == {'missing': 1, 'rewritten': 1, 'unchanged': 2}
    assert not (target / 'target.first.rs').exists()
    hw.generate(data, root, write=True)
    assert (target / 'target.first.rs').read_text() == sizes.instantiate(SOURCE, fixture(8))
    assert not (target / 'Prover.first.toml').exists()
    assert hw.generate(data, root, write=True)['counts'] == report['counts']
    (target / 'target.first.rs').write_text('do not overwrite')
    assert hw.generate(data, root, write=True)['counts']['not_scalable'] == 1
    assert (target / 'target.first.rs').read_text() == 'do not overwrite'


def test_warnings_require_review(tmp_path, monkeypatch, sizes):
    # Reuse the real integration setup with warnings to verify fail-closed writes.
    sizes.check = lambda *_: ([], ['literal index width'])
    root, data = tmp_path / 'root', tmp_path / 'data'
    bench = root / 'benchmarking'
    original = bench / 'datasets/demo'
    original.mkdir(parents=True)
    (bench / 'datasets/case_sets').mkdir()
    target = data / 'cap32/demo'
    target.mkdir(parents=True)
    (original / 'target.f.rs').write_text(SOURCE)
    for folder, n in [(original, 4), (target, 8)]:
        (folder / 'fixture.json').write_text(json.dumps(fixture(n)))
    (bench / 'datasets/case_sets/npbench_pythran_perf.json').write_text(json.dumps({'cases': [
        {'case': 'demo__f', 'function': 'f', 'path': 'datasets/demo'}]}))
    (data / 'points.json').write_text(json.dumps({'points': [
        {'case': 'demo__f', 'cap': 32, 'status': 'scaled', 'folder': 'cap32/demo', 'fixture': 'fixture.json'}]}))
    monkeypatch.setattr(hw, 'load_sizes', lambda _: sizes)
    report = hw.generate(data, root, write=True)
    assert report['counts'] == {'needs_review': 1, 'missing': 1}
    assert not (target / 'target.f.rs').exists()


def test_count_failures_resume_only_matching_artifacts(tmp_path, monkeypatch):
    from benchmarking.scaling_npb import handwritten_counts as counts
    monkeypatch.setenv('HW_CHECK_LOCK', str(tmp_path / 'test-heavy.lock'))
    root = tmp_path / 'root'
    folder = root / 'data/demo'
    folder.mkdir(parents=True)
    source = folder / 'target.f.rs'
    source.write_text('scaled circuit')
    (folder / 'fixture.json').write_text('{}')
    binary = root / 'halo2_hw'
    binary.write_text('binary')
    audit = root / 'audit.json'
    audit.write_text(json.dumps({'baseline_root': str(root), 'points': [
        {'case': 'demo__f_cap32', 'cap': 32, 'status': 'rewritten', 'system': 'halo2_hw',
         'output': str(source), 'output_sha256': counts.digest(source)}]}))
    args = SimpleNamespace(audit=audit, cap=32, cases=None, halo2_bin=binary,
                           memory_runner=root / 'memrun.sh', memory='8G', timeout=30,
                           work=root / 'work', out=root / 'counts.jsonl')
    calls = []
    def invoke(cmd, *_):
        calls.append(cmd)
        assert 'stats' in cmd and 'prove' not in cmd
        Path(cmd[cmd.index('--out') + 1]).write_text(json.dumps({'ok': True, 'cells': 12}))
        return {'status': 'ok', 'returncode': 0}
    monkeypatch.setattr(counts.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({'sources': {'demo__f': 'scaled circuit'}})))
    monkeypatch.setattr(counts, 'invoke', invoke)
    assert counts.main(args) == 0
    assert counts.main(args) == 0
    assert len(calls) == 1
    binary.write_text('changed binary')
    assert counts.main(args) == 0
    assert len(calls) == 2
    monkeypatch.setattr(counts, 'invoke', lambda *_: {'status': 'timeout'})
    binary.write_text('another binary')
    assert counts.main(args) == 1
    assert json.loads(args.out.read_text().splitlines()[-1])['status'] == 'timeout'
    source.write_text('tampered source')
    assert counts.main(args) == 1
    assert json.loads(args.out.read_text().splitlines()[-1])['status'] == 'source_mismatch'


def test_summary_overrides_per_target_preserves_failures(tmp_path):
    from benchmarking.scaling_npb.handwritten_summary import summarize
    original, corrected, counts = [tmp_path / x for x in ['old.json', 'new.json', 'hw.jsonl']]
    original.write_text(json.dumps({'cap32': {'x_cap32': {'halo2': {'cells': 60}, 'noir': {'gates': 20}}}}))
    corrected.write_text(json.dumps({'cap32': {'x_cap32': {'halo2': {'cells': 30}}}}))
    counts.write_text('\n'.join(json.dumps(x) for x in [
        {'case': 'x_cap32', 'system': 'halo2_hw', 'status': 'ok', 'output_sha256': 'a',
         'result': {'cells': {'advice': 20, 'lookup': 10}}},
        {'case': 'x_cap32', 'system': 'noir_hw', 'status': 'timeout', 'output_sha256': 'b'}]))
    audit = {'points': [
        {'case': 'x_cap32', 'cap': 32, 'system': 'halo2_hw', 'status': 'rewritten', 'output_sha256': 'a'},
        {'case': 'x_cap32', 'cap': 32, 'system': 'noir_hw', 'status': 'rewritten', 'output_sha256': 'b'},
        {'case': 'y_cap32', 'cap': 32, 'system': 'halo2_hw', 'status': 'rewritten', 'output_sha256': 'c'}]}
    report = summarize(audit, [counts], [original, corrected])
    assert report['points'][0]['ratio'] == 1
    assert report['points'][1]['status'] == 'timeout'
    assert report['points'][2]['status'] == 'not_run'
    assert report['groups'][0]['comparable'] == 1
    audit['points'][0]['output_sha256'] = 'changed'
    assert summarize(audit, [counts], [original])['points'][0]['status'] == 'stale_source'
