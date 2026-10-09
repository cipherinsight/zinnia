// npbench_polybench_syr2k_syr2k__kernel (NPBench polybench/syr2k), Noir baseline.
// Kernel: for each row i of C: Float[N, N], C[i, :i + 1] *= beta, then for
// each k < M, C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k] +
// B[:i + 1, k] * alpha * A[i, k], with A, B: Float[N, M], in place; the output
// is C after the call (`mutated_outputs`; its upper triangle is the input's).
// NumPy's order (NPB_BASELINES.md section 5.3), left to right: every
// A[j, k] * alpha and B[j, k] * alpha is rounded (the same value for every
// row i, so computed once), then each product with B[i, k] or A[i, k] is
// rounded; the updates add exactly.
// @outputs: float[N * N]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 13; // = A_0
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
    let mut a_alpha: [[Field; M]; N] = [[0; M]; N];
    let mut b_alpha: [[Field; M]; N] = [[0; M]; N];
    for j in 0..N {
        for k in 0..M {
            a_alpha[j][k] = mul(A[j][k], alpha);
            b_alpha[j][k] = mul(B[j][k], alpha);
        }
    }
    let mut c = C;
    for i in 0..N {
        for j in 0..i + 1 {
            c[i][j] = mul(c[i][j], beta);
        }
        for k in 0..M {
            for j in 0..i + 1 {
                c[i][j] += mul(a_alpha[j][k], B[i][k]) + mul(b_alpha[j][k], A[i][k]);
            }
        }
    }
    c
}
