// npbench_polybench_gesummv_gesummv__kernel (NPBench polybench/gesummv), Noir baseline.
// Kernel: alpha * A @ x + beta * B @ x on A, B: Float[M, N], x: Float[N].
// `*` and `@` group left to right, so each term is (alpha * A) @ x: every
// element of alpha * A is a rounded product, then each row's dot product with
// x is one reduction, rescaled once.
// @outputs: float[M]

use noir_hw::{check_input, check_inputs, dot, mul};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 16; // = A_0
global N: u32 = 16; // = A_1

// (factor * row) @ x with each product of `factor * row` rounded.
fn scaled_dot(factor: Field, row: [Field; N], x: [Field; N]) -> Field {
    let mut scaled: [Field; N] = [0; N];
    for j in 0..N {
        scaled[j] = mul(factor, row[j]);
    }
    dot(scaled, x)
}

fn main(alpha: Field, beta: Field, A: [[Field; N]; M], B: [[Field; N]; M], x: [Field; N]) -> pub [Field; M] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    check_inputs(x);
    let mut result: [Field; M] = [0; M];
    for i in 0..M {
        check_inputs(A[i]);
        check_inputs(B[i]);
        result[i] = scaled_dot(alpha, A[i], x) + scaled_dot(beta, B[i], x);
    }
    result
}
