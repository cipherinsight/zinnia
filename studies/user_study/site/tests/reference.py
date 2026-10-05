"""Test-only solutions. Never served or mounted into participant sandboxes."""
from ...tasks import TASKS
from ..content import names

PYTHON = {t.name: 'def kernel(' + ', '.join(names(t.name)) + '):\n' + '\n'.join('    '+line for line in next(iter(t.variants.values())).replace('{R}', '3').splitlines()) + '\n' for t in TASKS}
PYTHON['practice'] = 'def kernel(x, y):\n    return x + y\n'
RUST = {
    'practice': 'fn kernel(x: &Array1<i64>, y: &Array1<i64>) -> Array1<i64> { x + y }',
    'calibrate': '''fn kernel(x: &Array1<i64>) -> Array1<i64> {
        x.mapv(|v| (2*v+5).clamp(0, 255))
    }''',
    'background': '''fn kernel(x: &Array2<i64>) -> Array2<i64> {
        let mut out = x.clone();
        for c in 0..x.ncols() {
            let low = *x.column(c).iter().min().unwrap();
            for t in 0..x.nrows() { out[[t,c]] -= low; }
        }
        out
    }''',
    'threshold_energy': '''fn kernel(x: &Array2<i64>, floor: &Array1<i64>) -> Array1<i64> {
        Array1::from_iter(x.rows().into_iter().map(|row| {
            row.iter().zip(floor).filter(|(v,f)| v > f).map(|(v,_)| v*v).sum()
        }))
    }''',
    'distances': '''fn kernel(x: &Array2<i64>, y: &Array2<i64>) -> Array2<i64> {
        Array2::from_shape_fn((x.nrows(),y.nrows()), |(i,j)| {
            x.row(i).iter().zip(y.row(j)).map(|(a,b)| (a-b)*(a-b)).sum()
        })
    }''',
    'pool': '''fn kernel(x: &Array4<i64>) -> Array4<i64> {
        let s = x.shape();
        Array4::from_shape_fn((s[0],s[1]/2,s[2]/2,s[3]), |(b,i,j,c)| {
            x[[b,2*i,2*j,c]] + x[[b,2*i+1,2*j,c]] +
            x[[b,2*i,2*j+1,c]] + x[[b,2*i+1,2*j+1,c]]
        })
    }''',
    'smooth': '''fn kernel(x: &Array2<i64>) -> Array2<i64> {
        let mut a = x.clone();
        for _ in 0..3 {
            let mut next = a.clone();
            for i in 1..a.nrows()-1 {
                for j in 1..a.ncols()-1 {
                    next[[i,j]] = (4*a[[i,j]] + a[[i-1,j]] + a[[i+1,j]] + a[[i,j-1]] + a[[i,j+1]]) / 8;
                }
            }
            a = next;
        }
        a
    }''',
}

from ...revision2.reference import PYTHON as PYTHON_V2, RUST as RUST_V2
PYTHON.update(PYTHON_V2)
RUST.update(RUST_V2)

from ...revision3.reference import PYTHON as PYTHON_V3, RUST as RUST_V3
PYTHON.update(PYTHON_V3)
RUST.update(RUST_V3)

from ...revision4.reference import PYTHON as PYTHON_V4, RUST as RUST_V4
PYTHON.update(PYTHON_V4)
RUST.update(RUST_V4)

# Matched matrix training; historical addition practice remains unchanged.
for task in ('practice_zinnia_v2', 'practice_rust_v2'):
    PYTHON[task] = '''def kernel(x, baseline):
    return np.maximum(x - baseline, 0).sum(axis=1)
'''
    RUST[task] = '''fn kernel(x: &Array2<i64>, baseline: &Array1<i64>) -> Array1<i64> {
    (x - baseline).mapv(|v| v.max(0)).sum_axis(Axis(1))
}'''

from ...revision5.reference import PYTHON as PYTHON_V5, RUST as RUST_V5
PYTHON.update(PYTHON_V5)
RUST.update(RUST_V5)
