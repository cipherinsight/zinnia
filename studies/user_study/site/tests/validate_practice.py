"""Exercise natural practice implementations through the participant engines."""
import json
import tempfile
from ..execution import Runner
from ..content import cases
from .reference import PYTHON, RUST


def main():
    variants = {
        'zinnia': [PYTHON['practice_zinnia_v2'], '''def kernel(x, baseline):
    out = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            d = x[i, j] - baseline[j]
            if d > 0:
                out[i] += d
    return out
''', '''def kernel(x, baseline):
    d = x - baseline
    return np.sum(np.where(d > 0, d, 0), axis=1)
'''],
        'rust': [RUST['practice_rust_v2'], '''fn kernel(x: &Array2<i64>, baseline: &Array1<i64>) -> Array1<i64> {
    let mut out = Array1::zeros(x.nrows());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let d = x[(i,j)] - baseline[j];
            if d > 0 { out[i] += d; }
        }
    }
    out
}'''],
    }
    report = []
    with tempfile.TemporaryDirectory() as runtime:
        runner = Runner(runtime)
        try:
            for condition, sources in variants.items():
                task = 'practice_'+condition+'_v2'
                inputs = cases(task) + [{'x': [[-512,4096],[4096,-512],[0,0]], 'baseline': [4096,-512]}]
                for index, source in enumerate(sources):
                    result = runner.execute(condition, task, source, inputs)
                    assert result['status'] == 'passed', result
                    assert result['engine'] == ('risc0-zkvm' if condition == 'rust' else 'zinnia-ir-mock'), result
                    report.append({'condition': condition, 'variant': index, 'cases': len(inputs), 'status': result['status'], 'engine': result['engine']})
                bad = sources[0].replace('np.maximum(x - baseline, 0).sum(axis=1)', 'np.maximum((x - baseline).sum(axis=1), 0)').replace('(x - baseline).mapv(|v| v.max(0)).sum_axis(Axis(1))', '(x - baseline).sum_axis(Axis(1)).mapv(|v| v.max(0))')
                result = runner.execute(condition, task, bad, inputs)
                assert result['status'] == 'failed', result
                report.append({'condition': condition, 'negative_control': 'clamp after sum', 'status': result['status']})
        finally:
            runner.pool.shutdown()
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
