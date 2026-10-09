// npbench_polybench_syr2k_syr2k__kernel, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source:
//   for i in range(N):
//       C[i, :i + 1] *= beta
//       for k in range(M):
//           C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k]
//                          + B[:i + 1, k] * alpha * A[i, k]
// C[i, :i + 1] *= beta is an elementwise array statement: a loop over
// j in 0..=i multiplying each element by beta (L5, "element-wise ops,
// broadcasting"). The second statement's right-hand side is, left to right,
// (A[:i+1,k] * alpha) * B[i,k] + (B[:i+1,k] * alpha) * A[i,k]: for each j,
// A[j, k] * alpha and B[j, k] * alpha are rounded, each multiplied again by
// the opposite row's scalar, the two products added, then added into
// C[i, j] (NPB_FAIRNESS.md item 3: products rounded before accumulation,
// additions exact). A[j, k] * alpha does not depend on i, but the source
// recomputes it fresh every time the statement executes (once per (i, k)
// for each j <= i, i.e. up to N - j times over the outer loop for a given
// (j, k)); a literal translation computes it the same number of times, not
// once (L3: no hoisting of loop-invariant work).
// @outputs: float[N * N]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 13; // = A_0 (C_0, C_1)
global M: u32 = 9; // = A_1

fn main(
    alpha: Field,
    beta: Field,
    C: [[Field; N]; N],
    A: [[Field; M]; N],
    B: [[Field; M]; N],
) -> pub [[Field; N]; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..N {
        check_inputs(C[i]);
        check_inputs(A[i]);
        check_inputs(B[i]);
    }
    let mut c = C;
    for i in 0..N {
        // C[i, :i + 1] *= beta
        for j in 0..i + 1 {
            c[i][j] = mul(c[i][j], beta);
        }
        // for k in range(M): C[i, :i + 1] += ...
        for k in 0..M {
            for j in 0..i + 1 {
                // A[j, k] * alpha * B[i, k]
                let a_alpha = mul(A[j][k], alpha);
                let term1 = mul(a_alpha, B[i][k]);
                // B[j, k] * alpha * A[i, k]
                let b_alpha = mul(B[j][k], alpha);
                let term2 = mul(b_alpha, A[i][k]);
                // C[i, j] += term1 + term2
                c[i][j] += term1 + term2;
            }
        }
    }
    c
}
