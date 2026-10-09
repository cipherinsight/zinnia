// npbench_polybench_gemm_gemm__kernel (NPBench polybench/gemm), Noir baseline.
// Kernel: `C[:] = alpha * A @ B + beta * C` on C: Float[NI, NJ],
// A: Float[NI, NK], B: Float[NK, NJ] and scalars alpha, beta, in place; the
// output is C after the call (`mutated_outputs`).
// NumPy's order: alpha * A is materialized (each element rounded), then @ B
// (each element one reduction, rescaled once), beta * C is rounded
// elementwise, and the sum is exact.
// @outputs: float[NI * NJ]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 14; // = C_0
global NJ: u32 = 15; // = C_1
global NK: u32 = 16; // = A_1

fn main(
    alpha: Field,
    beta: Field,
    C: [[Field; NJ]; NI],
    A: [[Field; NK]; NI],
    B: [[Field; NJ]; NK],
) -> pub [[Field; NJ]; NI] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..NI {
        check_inputs(C[i]);
        check_inputs(A[i]);
    }
    for k in 0..NK {
        check_inputs(B[k]);
    }
    let mut result: [[Field; NJ]; NI] = [[0; NJ]; NI];
    for i in 0..NI {
        let mut alpha_a: [Field; NK] = [0; NK];
        for k in 0..NK {
            alpha_a[k] = mul(alpha, A[i][k]);
        }
        for j in 0..NJ {
            let mut acc: Field = 0;
            for k in 0..NK {
                acc += alpha_a[k] * B[k][j];
            }
            result[i][j] = rescale(acc) + mul(beta, C[i][j]);
        }
    }
    result
}
