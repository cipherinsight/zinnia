// pythran_deuxd_convolution__conv: literal translation of kernel.py (LITERAL_BASELINES.md).
// Clamp indices are compile-time numbers; retain all four source loops and round every product.
// @outputs: float[SX0 * SX1]

use noir_hw::{check_input, check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global SX0: u32 = 16; // = x_0
global SX1: u32 = 16; // = x_1
global SW0: u32 = 5; // = weights_0
global SW1: u32 = 5; // = weights_1

// The kernel's clamp(i, offset - SW0 // 2, SW0), on loop constants.
fn clamp(i: u32, offset: u32) -> u32 {
    let shifted = i + offset;
    let low = if shifted < SW0 / 2 { 0 } else { shifted - SW0 / 2 };
    if low > SW0 { SW0 } else { low }
}

fn main(x: [[Field; SX1]; SX0], weights: [[Field; SW1]; SW0]) -> pub [[Field; SX1]; SX0] {
    // Input domain: every element read is a Q32 code in the window.
    for r in 0..SW0 + 1 {
        for c in 0..SW0 + 1 {
            check_input(x[r][c]);
        }
    }
    for ii in 0..SW0 {
        check_inputs(weights[ii]);
    }
    let mut result = [[0; SX1]; SX0];
    for i in 0..SX0 { for j in 0..SX1 { for ii in 0..SW0 { for jj in 0..SW1 {
        let idx = (clamp(i,ii),clamp(j,jj));
        result[i][j] += mul(x[idx.0][idx.1],weights[ii][jj]);
    } } } }
    result
}
