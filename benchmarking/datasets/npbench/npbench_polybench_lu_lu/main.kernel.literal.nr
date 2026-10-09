// npbench_polybench_lu_lu__kernel, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// LU decomposition without pivoting of A: Float[N, N] in place (Doolittle,
// row by row); the output is A after the call (mutated_outputs). Source:
//   for i in range(N):
//       for j in range(i):
//           A[i, j] -= A[i, :j] @ A[:j, j]
//           A[i, j] /= A[j, j]
//       for j in range(i, N):
//           A[i, j] -= A[i, :i] @ A[:i, j]
// Both @ calls are single NumPy reductions (one rescale each, L5); an empty
// slice (j == 0 or i == 0) makes the dot product's defining sum empty, i.e.
// 0, so the preceding subtraction is skipped rather than computed as "minus
// zero" (the same value either way, no extra gate). The two assignments to
// A[i, j] in the j < i branch are two source statements executed in order:
// the second reads the value the first just wrote, as the source does; this
// is not common-subexpression elimination, there is nothing recomputed. `/`
// uses the division gadget the arithmetic contract shares across baselines
// (canonical_div_nonzero): the dividend after subtraction is not bounded to
// signed64 by this point in the recurrence, which only this gadget's domain
// covers.
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
            let mut after_sub = a[i][j];
            if j > 0 {
                let mut acc: Field = 0;
                for k in 0..j {
                    acc += a[i][k] * a[k][j];
                }
                after_sub -= rescale(acc);
            }
            // A[i, j] /= A[j, j]
            a[i][j] = canonical_div_nonzero(after_sub, a[j][j]);
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
    }
    a
}
