// npbench_polybench_lu_lu__kernel, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Each @ is a dot with one rescale; empty dots are constant zero.
// Subtraction remains explicit even for an empty dot. Division uses the
// shared canonical_nonzero domain, as in the optimized statement.

// @outputs: float[N * N]

use noir_hw::{canonical_div_nonzero, check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 15; // = A_0

fn main(A: [[Field; N]; N]) -> pub [[Field; N]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(A[i]);
    }
    let mut a = A;
    for i in 0..N {
        for j in 0..i {
            // A[i, j] -= A[i, :j] @ A[:j, j]
            let mut acc: Field = 0;
            for k in 0..j {
                acc += a[i][k] * a[k][j];
            }
            let product = if j == 0 { 0 } else { rescale(acc) };
            let after_sub = a[i][j] - product;
            // A[i, j] /= A[j, j]
            a[i][j] = canonical_div_nonzero(after_sub, a[j][j]);
        }
        for j in i..N {
            // A[i, j] -= A[i, :i] @ A[:i, j]
            let mut acc: Field = 0;
            for k in 0..i {
                acc += a[i][k] * a[k][j];
            }
            let product = if i == 0 { 0 } else { rescale(acc) };
            a[i][j] -= product;
        }
    }
    a
}
