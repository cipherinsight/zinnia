// npbench_polybench_ludcmp_ludcmp__kernel (NPBench polybench/ludcmp), Noir baseline.
// Kernel: LU decomposition without pivoting of `A: Float[N, N]` (as
// polybench/lu), then L y = b by forward substitution
// (y[i] = b[i] - A[i, :i] @ y[:i]) and U x = y by back substitution
// (x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]); returns (x, y).
// A is written but not returned, so it is no output.
// Each @ is one reduction, rescaled once (an empty one is 0). Every diagonal
// element divides, with either sign: its magnitude is proven once
// (`abs_int64`) and every quotient by it reuses it (`div_by_abs`).
// @outputs: float[N]; float[N]

use noir_hw::{abs_int64, check_inputs, div_by_abs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 15; // = A_0

fn main(A: [[Field; N]; N], b: [Field; N]) -> pub ([Field; N], [Field; N]) {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(A[i]);
    }
    check_inputs(b);
    let mut a = A;
    let mut diagonal_abs: [Field; N] = [0; N];
    for i in 0..N {
        for j in 0..i {
            // A[i, j] = (A[i, j] - A[i, :j] @ A[:j, j]) / A[j, j]
            let mut numerator = a[i][j];
            if j > 0 {
                let mut acc: Field = 0;
                for k in 0..j {
                    acc += a[i][k] * a[k][j];
                }
                numerator -= rescale(acc);
            }
            a[i][j] = div_by_abs(numerator, a[j][j], diagonal_abs[j]);
        }
        for j in i..N {
            // A[i, j] -= A[i, :i] @ A[:i, j]
            if i > 0 {
                let mut acc: Field = 0;
                for k in 0..i {
                    acc += a[i][k] * a[k][j];
                }
                a[i][j] -= rescale(acc);
            }
        }
        diagonal_abs[i] = abs_int64(a[i][i]);
    }
    // y[i] = b[i] - A[i, :i] @ y[:i]
    let mut y: [Field; N] = [0; N];
    for i in 0..N {
        y[i] = b[i];
        if i > 0 {
            let mut acc: Field = 0;
            for k in 0..i {
                acc += a[i][k] * y[k];
            }
            y[i] -= rescale(acc);
        }
    }
    // x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i], from the last row up
    let mut x: [Field; N] = [0; N];
    for t in 0..N {
        let i = N - 1 - t;
        let mut numerator = y[i];
        if i + 1 < N {
            let mut acc: Field = 0;
            for k in i + 1..N {
                acc += a[i][k] * x[k];
            }
            numerator -= rescale(acc);
        }
        x[i] = div_by_abs(numerator, a[i][i], diagonal_abs[i]);
    }
    (x, y)
}
