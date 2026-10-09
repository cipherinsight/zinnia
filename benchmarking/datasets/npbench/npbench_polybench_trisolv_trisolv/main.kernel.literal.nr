// npbench_polybench_trisolv_trisolv__kernel (NPBench polybench/trisolv), Noir
// literal baseline. LITERAL_BASELINES.md: direct translation of kernel.py,
// no optimization.
//
// Source (kernel.py):
//   for i in range(x.shape[0]):
//       x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
// Translation choices:
// - The loop runs in source order, one circuit statement per iteration; `x`
//   is overwritten before any later statement reads it and the strict upper
//   triangle of `L` is never read by any iteration, so neither is checked.
// - `L[i, :i] @ x[:i]` (L5 table, single `@` call): accumulated with one
//   `rescale` at the end, matching how NumPy rounds `@` once; `i = 0` is the
//   constant `b[0]` the empty dot product always reduces to.
// - `/ L[i, i]`: a single plain division of the subtraction's result by the
//   diagonal element, via the general `div`, which takes the divisor's sign
//   and magnitude from itself (not a magnitude reused from elsewhere, so
//   not the forbidden `div_by_abs` pattern); this is already the cheapest
//   correct form here, so the literal and optimized Noir baselines compute
//   the same circuit.
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
