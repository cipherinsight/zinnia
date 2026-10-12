"""Final-compiler run (2569c9ed) only: two observational probes no longer compile, although the
original programs do. Both programs have only fixed shapes, so they have no bounded arrays.
Evidence: their census layouts have no dynamic part, and Task A's IR-generator instrumentation
on the same compiler (paper-numbers/A/final/classification.json) records no runtime-shaped
DynamicNDArray value for them (runtime_shape false, dynamic_type false)."""
import measure, json
from pathlib import Path
# Task A's final-compiler output: $RUN/paper-numbers-final/A on the remote, paper-numbers/A/final here.
HERE = Path(__file__).resolve().parent
A = next(p for p in [HERE.parent / 'A' / 'classification.json', HERE.parent.parent / 'A' / 'final' / 'classification.json'] if p.exists())
cases = {
 'npbench_polybench_durbin_durbin': 'Probe rejected by the final compiler: the trace keeps views y[:k] that the loop modifies afterwards, which the compiler refuses to read (NumPy view semantics). All shapes are fixed: k ranges over range(1, r.shape[0]) with r of fixed shape 16, so every slice r[:k], y[:k] and np.flip of it has a static length.',
 'pythran_multitype': 'Probe of check2 crashes the final compiler (IRBuilder SelectI, arg without stmt_id) because the probe returns lists of different structure from the two branches. The only array, x + x, has the fixed shape of x (1,).',
}
klass = {r['id']: r for r in json.loads(A.read_text())}
for p in measure.entries():
    name = measure.ident(p)
    if p.program not in cases or (p.program == 'pythran_multitype' and p.function != 'check2'):
        continue
    r = json.loads((measure.HERE / 'probes' / (name + '.json')).read_text())
    if 'error' not in r:
        continue
    c = json.loads((measure.HERE / 'census' / (name + '.json')).read_text())
    assert not any('dynamic' in x for x in c['layout'])
    assert not klass[name]['runtime_shape'] and not klass[name]['dynamic_type'], name
    r['probe_error'] = r.pop('error'); r['dynamic'] = []; r['layout'] = []
    r['repair'] = cases[p.program]
    (measure.HERE / 'probes' / (name + '.json')).write_text(json.dumps(r, indent=2))
    print(name, 'audited: no runtime-shaped arrays')
