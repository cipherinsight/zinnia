// npbench_polybench_gesummv_gesummv__kernel, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// `return alpha * A @ x + beta * B @ x` on A, B: Float[M, N], x: Float[N].
// Python's `*` and `@` share one precedence level and group left to right,
// so the source expression is ((alpha * A) @ x) + ((beta * B) @ x):
// alpha * A is an elementwise array (every element a rounded product), then
// each row's dot product with x is one NumPy reduction, rescaled once (L5);
// the two terms are added elementwise.
// @outputs: float[M]

use noir_hw::{check_input, check_inputs, dot, mul};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 16; // = A_0
global N: u32 = 16; // = A_1

fn main(alpha: Field, beta: Field, A: [[Field; N]; M], B: [[Field; N]; M], x: [Field; N]) -> pub [Field; M] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    check_inputs(x);
    let mut result: [Field; M] = [0; M];
    for i in 0..M {
        check_inputs(A[i]);
        check_inputs(B[i]);
        // alpha * A[i, :]
        let mut scaled_a: [Field; N] = [0; N];
        for j in 0..N {
            scaled_a[j] = mul(alpha, A[i][j]);
        }
        // (alpha * A[i, :]) @ x
        let first = dot(scaled_a, x);
        // beta * B[i, :]
        let mut scaled_b: [Field; N] = [0; N];
        for j in 0..N {
            scaled_b[j] = mul(beta, B[i][j]);
        }
        // (beta * B[i, :]) @ x
        let second = dot(scaled_b, x);
        result[i] = first + second;
    }
    result
}
