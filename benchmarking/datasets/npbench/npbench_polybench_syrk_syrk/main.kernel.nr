// npbench_polybench_syrk_syrk__kernel (NPBench polybench/syrk), Noir baseline.
// Kernel: for each row i of C: Float[N, N], C[i, :i + 1] *= beta, then for
// each k < M, C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k], with
// A: Float[N, M], in place; the output is C after the call
// (`mutated_outputs`; its upper triangle is the input's).
// NumPy's order (NPB_BASELINES.md section 5.3): alpha * A[i, k] is a rounded
// scalar, computed once per (i, k); each element's product with it is
// rounded; the updates add exactly.
// @outputs: float[N * N]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 14; // = A_0
global M: u32 = 10; // = A_1

fn main(alpha: Field, beta: Field, C: [[Field; N]; N], A: [[Field; M]; N]) -> pub [[Field; N]; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..N {
        check_inputs(C[i]);
        check_inputs(A[i]);
    }
    let mut c = C;
    for i in 0..N {
        for j in 0..i + 1 {
            c[i][j] = mul(c[i][j], beta);
        }
        for k in 0..M {
            let scale = mul(alpha, A[i][k]);
            for j in 0..i + 1 {
                c[i][j] += mul(scale, A[j][k]);
            }
        }
    }
    c
}
