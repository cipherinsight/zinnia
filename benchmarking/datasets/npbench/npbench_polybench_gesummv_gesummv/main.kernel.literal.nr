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

fn main(
    alpha: Field,
    beta: Field,
    A: [[Field; N]; M],
    B: [[Field; N]; M],
    x: [Field; N],
) -> pub [Field; M] {
    // Input domain: every input element is a Q32 code in the window.
    check_input(alpha);
    check_input(beta);
    check_inputs(x);
    for i in 0..M {
        check_inputs(A[i]);
        check_inputs(B[i]);
    }
    let mut scaled_a = [[0; N]; M];
    for i in 0..M {
        for j in 0..N {
            scaled_a[i][j] = mul(alpha, A[i][j]);
        }
    }
    let mut first = [0; M];
    for i in 0..M {
        first[i] = dot(scaled_a[i], x);
    }
    let mut scaled_b = [[0; N]; M];
    for i in 0..M {
        for j in 0..N {
            scaled_b[i][j] = mul(beta, B[i][j]);
        }
    }
    let mut second = [0; M];
    for i in 0..M {
        second[i] = dot(scaled_b[i], x);
    }
    let mut result = [0; M];
    for i in 0..M {
        result[i] = first[i] + second[i];
    }
    result
}
