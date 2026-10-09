// npbench_polybench_cholesky_cholesky__kernel (NPBench polybench/cholesky), Noir baseline.
// Kernel: Cholesky factorization of `A: Float[N, N]` in place, row by row:
// A[i, j] = (A[i, j] - A[i, :j] @ A[j, :j]) / A[j, j] for j < i, then
// A[i, i] = sqrt(A[i, i] - A[i, :i] @ A[i, :i]); the output is all of A after
// the call (`mutated_outputs`), the untouched upper triangle included.
// Each np.dot is one reduction, rescaled once (an empty one is 0). The
// divisors are square roots, non-negative by construction, so the quotients
// are `divmod_pos` (a zero divisor has no witness, as NumPy's result is not
// finite; a negative radicand has none either, as NumPy's is NaN).
// @outputs: float[N * N]

use noir_hw::{check_inputs, divmod_pos, rescale, sqrt, SCALE};

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
            // A[i, j] = (A[i, j] - A[i, :j] @ A[j, :j]) / A[j, j]
            let mut numerator = a[i][j];
            if j > 0 {
                let mut acc: Field = 0;
                for k in 0..j {
                    acc += a[i][k] * a[j][k];
                }
                numerator -= rescale(acc);
            }
            let (quotient, _) = divmod_pos(numerator * SCALE, a[j][j]);
            a[i][j] = quotient;
        }
        // A[i, i] = sqrt(A[i, i] - A[i, :i] @ A[i, :i])
        let mut radicand = a[i][i];
        if i > 0 {
            let mut acc: Field = 0;
            for k in 0..i {
                acc += a[i][k] * a[i][k];
            }
            radicand -= rescale(acc);
        }
        a[i][i] = sqrt(radicand);
    }
    a
}
