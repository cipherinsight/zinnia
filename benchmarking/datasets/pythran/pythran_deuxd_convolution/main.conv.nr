// pythran_deuxd_convolution__conv (Pythran deuxd_convolution), Noir baseline.
// Kernel: on x: Float[SX0, SX1], weights: Float[SW0, SW1],
// result[i, j] = sum_{ii, jj} x[r(i, ii), c(j, jj)] * weights[ii, jj] with the
// kernel's clamp r(i, ii) = min(max(0, i + ii - SW0 // 2), SW0) and the same
// for columns, also clamped to SW0 (the kernel passes sw[0] for both). The
// indices are loop constants. Each output accumulates its products into one
// variable from 0, one reduction rescaled once (NPB_BASELINES.md section 5,
// rule 3).
// Only x[:SW0 + 1, :SW0 + 1] is read, and only it is checked. From row
// SW0 + SW0 // 2 on, every row index clamps to SW0, so those rows of result
// repeat that row's computation, and the same holds for columns: each
// distinct computation (ROW_KINDS * COL_KINDS of them) is done once and output
// wherever it occurs.
// @outputs: float[SX0 * SX1]

use noir_hw::{check_input, check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global SX0: u32 = 16; // = x_0
global SX1: u32 = 16; // = x_1
global SW0: u32 = 5; // = weights_0
global SW1: u32 = 5; // = weights_1
// Rows and columns of result with distinct index patterns: the clamp
// saturates from SW0 + SW0 // 2 on.
global ROW_KINDS: u32 = 8; // = min(x_0, weights_0 + weights_0 // 2 + 1)
global COL_KINDS: u32 = 8; // = min(x_1, weights_0 + weights_0 // 2 + 1)

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
    let mut kinds: [[Field; COL_KINDS]; ROW_KINDS] = [[0; COL_KINDS]; ROW_KINDS];
    for i in 0..ROW_KINDS {
        for j in 0..COL_KINDS {
            let mut acc: Field = 0;
            for ii in 0..SW0 {
                for jj in 0..SW1 {
                    acc += x[clamp(i, ii)][clamp(j, jj)] * weights[ii][jj];
                }
            }
            kinds[i][j] = rescale(acc);
        }
    }
    let mut result: [[Field; SX1]; SX0] = [[0; SX1]; SX0];
    for i in 0..SX0 {
        for j in 0..SX1 {
            let ki = if i < ROW_KINDS { i } else { ROW_KINDS - 1 };
            let kj = if j < COL_KINDS { j } else { COL_KINDS - 1 };
            result[i][j] = kinds[ki][kj];
        }
    }
    result
}
