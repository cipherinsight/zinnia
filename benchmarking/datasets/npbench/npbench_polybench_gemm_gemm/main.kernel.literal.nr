// npbench_polybench_gemm_gemm__kernel, Noir literal baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source: C[:] = alpha * A @ B + beta * C on C: Float[NI, NJ],
// A: Float[NI, NK], B: Float[NK, NJ], scalars alpha, beta.
// Translation choices:
// - NumPy's order: alpha * A is a fresh array, one rounded product per
//   element; (alpha * A) @ B is one reduction per output element, rescaled
//   once (accumulate unscaled products, rescale once); beta * C is a fresh
//   array, one rounded product per element; the final + is exact. No fusing
//   of the two * with the @ or the +.
// - The output is C after the call (`mutated_outputs`).
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
    // alpha * A (fresh array, one rounded product per element)
    let mut alpha_a: [[Field; NK]; NI] = [[0; NK]; NI];
    for i in 0..NI {
        for k in 0..NK {
            alpha_a[i][k] = mul(alpha, A[i][k]);
        }
    }
    let mut result: [[Field; NJ]; NI] = [[0; NJ]; NI];
    for i in 0..NI {
        for j in 0..NJ {
            // (alpha * A) @ B: one reduction per output element, rescaled once.
            let mut acc: Field = 0;
            for k in 0..NK {
                acc += alpha_a[i][k] * B[k][j];
            }
            // + beta * C
            result[i][j] = rescale(acc);
        }
    }
    // beta * C (fresh array, one rounded product per element)
    let mut beta_c: [[Field; NJ]; NI] = [[0; NJ]; NI];
    for i in 0..NI {
        for j in 0..NJ {
            beta_c[i][j] = mul(beta, C[i][j]);
        }
    }
    for i in 0..NI {
        for j in 0..NJ {
            result[i][j] = result[i][j] + beta_c[i][j];
        }
    }
    result
}
