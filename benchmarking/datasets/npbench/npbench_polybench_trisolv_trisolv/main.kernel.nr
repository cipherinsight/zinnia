// npbench_polybench_trisolv_trisolv__kernel (NPBench polybench/trisolv), Noir baseline.
// Kernel: forward substitution x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i] for
// i in order, on L: Float[N, N], x, b: Float[N], in place; the output is x
// after the call (`mutated_outputs`).
// Each L[i, :i] @ x[:i] is one reduction, rescaled once (the empty one for
// i = 0 is 0). The input x is overwritten before it is read and the strict
// upper triangle of L is never read, so neither is checked. `div` takes the
// divisor's sign from its window, shared with its input check (4,046 gates,
// against 4,094 with `abs_int64` and `div_by_abs`).
// @outputs: float[N]

use noir_hw::{check_input, check_inputs, div, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = L_0

fn main(L: [[Field; N]; N], x: [Field; N], b: [Field; N]) -> pub [Field; N] {
    // Input domain: b and the lower triangle of L are Q32 codes in the window.
    check_inputs(b);
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        let mut acc: Field = 0;
        for j in 0..i {
            check_input(L[i][j]);
            acc += L[i][j] * result[j];
        }
        check_input(L[i][i]);
        let rest = if i == 0 { b[0] } else { b[i] - rescale(acc) };
        result[i] = div(rest, L[i][i]);
    }
    let _ = x;
    result
}
