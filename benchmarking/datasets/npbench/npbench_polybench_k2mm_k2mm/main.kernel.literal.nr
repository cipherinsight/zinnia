// npbench_polybench_k2mm_k2mm__kernel, Noir literal baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source: D[:] = alpha * A @ B @ C + beta * D on A: Float[NI, NK],
// B: Float[NK, NJ], C: Float[NJ, NL], D: Float[NI, NL], scalars alpha, beta.
// Translation choices:
// - @ is left-associative, so NumPy evaluates ((alpha * A) @ B) @ C:
//   alpha * A is a fresh array (one rounded product per element), each @ is
//   one reduction per output element (rescaled once), left to right;
//   beta * D is a fresh array (one rounded product per element); the final +
//   is exact. No fusing of the two *, either @, or the final +.
// - The output is D after the call (`mutated_outputs`).
// @outputs: float[NI * NL]

use noir_hw::{check_input, check_inputs, mul, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NI: u32 = 14; // = A_0
global NK: u32 = 15; // = A_1
global NJ: u32 = 15; // = B_1
global NL: u32 = 16; // = C_1

fn main(
    alpha: Field,
    beta: Field,
    A: [[Field; NK]; NI],
    B: [[Field; NJ]; NK],
    C: [[Field; NL]; NJ],
    D: [[Field; NL]; NI],
) -> pub [[Field; NL]; NI] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    for i in 0..NI {
        check_inputs(A[i]);
        check_inputs(D[i]);
    }
    for k in 0..NK {
        check_inputs(B[k]);
    }
    for j in 0..NJ {
        check_inputs(C[j]);
    }
    // alpha * A (fresh array, one rounded product per element)
    let mut alpha_a: [[Field; NK]; NI] = [[0; NK]; NI];
    for i in 0..NI {
        for k in 0..NK {
            alpha_a[i][k] = mul(alpha, A[i][k]);
        }
    }
    // (alpha * A) @ B (one reduction per output element, rescaled once)
    let mut ab: [[Field; NJ]; NI] = [[0; NJ]; NI];
    for i in 0..NI {
        for j in 0..NJ {
            let mut acc: Field = 0;
            for k in 0..NK {
                acc += alpha_a[i][k] * B[k][j];
            }
            ab[i][j] = rescale(acc);
        }
    }
    // ((alpha * A) @ B) @ C (one reduction per output element, rescaled once)
    let mut abc: [[Field; NL]; NI] = [[0; NL]; NI];
    for i in 0..NI {
        for l in 0..NL {
            let mut acc: Field = 0;
            for j in 0..NJ {
                acc += ab[i][j] * C[j][l];
            }
            abc[i][l] = rescale(acc);
        }
    }
    // beta * D, then final addition in separate full passes.
    let mut beta_d = [[0; NL]; NI];
    for i in 0..NI { for l in 0..NL { beta_d[i][l] = mul(beta, D[i][l]); } }
    let mut result: [[Field; NL]; NI] = [[0; NL]; NI];
    for i in 0..NI {
        for l in 0..NL {
            result[i][l] = abc[i][l] + beta_d[i][l];
        }
    }
    result
}
