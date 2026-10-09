// trisolv: literal translation of kernel.py (LITERAL_BASELINES.md).
// Empty dot is constant zero; nonempty dot is rescaled once; ordinary division.
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
        let dot = if i == 0 { 0 } else { rescale(acc) };
        let rest = b[i] - dot;
        result[i] = div(rest, L[i][i]);
    }
    let _ = x;
    result
}
