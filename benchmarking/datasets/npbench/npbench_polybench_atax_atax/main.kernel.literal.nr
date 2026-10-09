// npbench_polybench_atax_atax__kernel, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// `return A @ x @ A` on A: Float[M, N], x: Float[N]. Python evaluates `@`
// left to right: y = A @ x (M dot products of a row of A with x), then
// y @ A (N dot products of y with a column of A). Each `@` is one NumPy
// reduction, so each dot product is a single term-by-term sum rescaled once
// (L5: "np.dot, @ ... the single rescale ... applies only to these
// one-call reductions").
// @outputs: float[N]

use noir_hw::{check_inputs, dot};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 13; // = A_0
global N: u32 = 16; // = A_1

fn main(A: [[Field; N]; M], x: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(A[i]);
    }
    check_inputs(x);

    // y = A @ x
    let mut y: [Field; M] = [0; M];
    for i in 0..M {
        y[i] = dot(A[i], x);
    }

    // return y @ A
    let mut result: [Field; N] = [0; N];
    for j in 0..N {
        let mut column: [Field; M] = [0; M];
        for i in 0..M {
            column[i] = A[i][j];
        }
        result[j] = dot(y, column);
    }
    result
}
