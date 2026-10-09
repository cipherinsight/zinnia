// npbench_polybench_ludcmp_ludcmp__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Dots include empty reductions. Shared canonical nonzero division preserves the optimized arithmetic/domain contract for derived pivots.
// @outputs: float[N]; float[N]

use noir_hw::{canonical_div_nonzero, check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 15; // = A_0

fn main(A: [[Field; N]; N], b: [Field; N]) -> pub ([Field; N], [Field; N]) {
    for i in 0..N {
        check_inputs(A[i]);
    }
    check_inputs(b);
    let mut a = A;
    let mut x = [0; N];
    let mut y = [0; N];
    for i in 0..N {
        for j in 0..i {
            let mut acc = 0;
            for k in 0..j {
                acc += a[i][k] * a[k][j];
            }
            a[i][j] -= rescale(acc);
            a[i][j] = canonical_div_nonzero(a[i][j], a[j][j]);
        }
        for j in i..N {
            let mut acc = 0;
            for k in 0..i {
                acc += a[i][k] * a[k][j];
            }
            a[i][j] -= rescale(acc);
        }
    }
    for i in 0..N {
        let mut acc = 0;
        for k in 0..i {
            acc += a[i][k] * y[k];
        }
        y[i] = b[i] - rescale(acc);
    }
    for t in 0..N {
        let i = N - 1 - t;
        let mut acc = 0;
        for k in i + 1..N {
            acc += a[i][k] * x[k];
        }
        x[i] = canonical_div_nonzero(y[i] - rescale(acc), a[i][i]);
    }
    (x, y)
}
