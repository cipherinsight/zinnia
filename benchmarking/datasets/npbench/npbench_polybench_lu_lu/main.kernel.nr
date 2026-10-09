// npbench_polybench_lu_lu__kernel (NPBench polybench/lu), Noir baseline.
// Kernel: LU decomposition without pivoting of `A: Float[N, N]` in place
// (Doolittle, row by row); the output is A after the call
// (`mutated_outputs`). Row i: for j < i,
// A[i, j] = (A[i, j] - A[i, :j] @ A[:j, j]) / A[j, j]; for j >= i,
// A[i, j] -= A[i, :i] @ A[:i, j].
// Each @ is one reduction, rescaled once (an empty one is 0). The divisor
// A[j, j] can have either sign; its magnitude is proven once (`abs_int64`)
// when row j is done, and every quotient by it reuses it (`div_by_abs`). The
// last diagonal element divides nothing, so it gets no such proof.
// @outputs: float[N * N]

use noir_hw::{abs_int64, check_inputs, div_by_abs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 15; // = A_0

fn main(A: [[Field; N]; N]) -> pub [[Field; N]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(A[i]);
    }
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
        if i + 1 < N {
            diagonal_abs[i] = abs_int64(a[i][i]);
        }
    }
    a
}
