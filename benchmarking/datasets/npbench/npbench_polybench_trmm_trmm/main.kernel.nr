// npbench_polybench_trmm_trmm__kernel (NPBench polybench/trmm), Noir baseline.
// Kernel: on alpha: Float, A: Float[NI, NI], B: Float[NI, NJ], in place:
// B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j]) for every i, j, then
// B *= alpha; the output is B after the call (`mutated_outputs`).
// Iteration i reads rows below i of B, which later iterations write, so every
// dot product reads the original B. Each is one reduction, rescaled once (the
// last row's is empty, 0.0); each element of B *= alpha is a rounded product.
// Only the strict lower triangle of A is read, and only it is checked.
// @outputs: float[NI * NJ]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 13; // = A_0
global NJ: u32 = 16; // = B_1

fn main(alpha: Field, A: [[Field; NI]; NI], B: [[Field; NJ]; NI]) -> pub [[Field; NJ]; NI] {
    // Input domain: every element read is a Q32 code in the window.
    check_input(alpha);
    for i in 0..NI {
        for k in 0..i {
            check_input(A[i][k]);
        }
        check_inputs(B[i]);
    }
    let mut result: [[Field; NJ]; NI] = [[0; NJ]; NI];
    for i in 0..NI {
        for j in 0..NJ {
            let mut updated = B[i][j];
            if i + 1 < NI {
                let mut acc: Field = 0;
                for k in i + 1..NI {
                    acc += A[k][i] * B[k][j];
                }
                updated += rescale(acc);
            }
            result[i][j] = mul(updated, alpha);
        }
    }
    result
}
