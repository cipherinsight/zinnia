// npbench_polybench_atax_atax__kernel (NPBench polybench/atax), Noir baseline.
// Kernel: `A @ x @ A` on A: Float[M, N], x: Float[N], evaluated left to
// right: y = A @ x, then y @ A.
// Each element of either product is one reduction, rescaled once.
// @outputs: float[N]

use noir_hw::{check_inputs, dot, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 13; // = A_0
global N: u32 = 16; // = A_1

fn main(A: [[Field; N]; M], x: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(A[i]);
    }
    check_inputs(x);
    let mut y: [Field; M] = [0; M];
    for i in 0..M {
        y[i] = dot(A[i], x);
    }
    let mut result: [Field; N] = [0; N];
    for j in 0..N {
        let mut acc: Field = 0;
        for i in 0..M {
            acc += y[i] * A[i][j];
        }
        result[j] = rescale(acc);
    }
    result
}
