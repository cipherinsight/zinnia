// npbench_polybench_mvt_mvt__kernel, Noir literal baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source:
//   x1 += A @ y_1
//   x2 += y_2 @ A
// on A: Float[N0, N1], x1: Float[N0], x2: Float[N1], y_1: Float[N1],
// y_2: Float[N0].
// Translation choices:
// - Each @ is one library call, so L5 gives it one rescale per output
//   element; += is exact addition onto the existing value, computed as its
//   own statement after the matching @.
// - The two statements are independent and computed in source order; the
//   outputs are x1 and x2 after the call (`mutated_outputs`).
// @outputs: float[N0]; float[N1]

use noir_hw::{check_inputs, dot, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N0: u32 = 16; // = A_0
global N1: u32 = 16; // = A_1

fn main(
    x1: [Field; N0],
    x2: [Field; N1],
    y_1: [Field; N1],
    y_2: [Field; N0],
    A: [[Field; N1]; N0],
) -> pub ([Field; N0], [Field; N1]) {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x1);
    check_inputs(x2);
    check_inputs(y_1);
    check_inputs(y_2);
    for i in 0..N0 {
        check_inputs(A[i]);
    }
    // x1 += A @ y_1
    let mut product1 = [0; N0];
    let mut new_x1 = x1;
    for i in 0..N0 {
        product1[i] = dot(A[i], y_1);
    }
    for i in 0..N0 {
        new_x1[i] += product1[i];
    }
    // x2 += y_2 @ A
    let mut product2 = [0; N1];
    let mut new_x2 = x2;
    for j in 0..N1 {
        let mut acc: Field = 0;
        for i in 0..N0 {
            acc += y_2[i] * A[i][j];
        }
        product2[j] = rescale(acc);
    }
    for j in 0..N1 {
        new_x2[j] += product2[j];
    }
    (new_x1, new_x2)
}
