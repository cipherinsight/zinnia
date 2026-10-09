// npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, Noir baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// `conv2d_bias` calls `conv2d(input, weights) + bias`; `conv2d`'s explicit
// Python loops are `for i in range(H_out): for j in range(W_out):`, with one
// array statement per (i, j):
//   output[:, i, j, :] = np.sum(
//       input[:, i:i + K, j:j + K, :, newaxis] * weights[newaxis, :, :, :],
//       axis=(1, 2, 3))
// The batch axis b and output-channel axis co are not explicit loop
// variables; they come from broadcasting the two operands against each
// other, so a literal translation loops over them in the array's own axis
// order (L5, "element-wise ops, broadcasting"). For each (b, i, j, co), the
// statement is np.sum(x * y) over the window and filter: one reduction,
// rescaled once (CONVENTIONS section 2). conv2d_bias's + bias is exact,
// elementwise over co (L5 broadcasting).
// This is the direct translation already: there is no redundant arithmetic
// in this kernel to eliminate, so it is structurally identical to the
// optimized baseline.
// @outputs: float[NB * H_OUT * W_OUT * C_OUT]

use noir_hw::{check_inputs, rescale};

// Sizes (hw_check.sizes): from fixture.json.
global NB: u32 = 4; // = input_0
global HEIGHT: u32 = 16; // = input_1
global WIDTH: u32 = 16; // = input_2
global C_IN: u32 = 2; // = input_3
global K: u32 = 2; // = weights_0
global C_OUT: u32 = 8; // = weights_3
global H_OUT: u32 = HEIGHT - K + 1;
global W_OUT: u32 = WIDTH - K + 1;

fn main(
    input: [[[[Field; C_IN]; WIDTH]; HEIGHT]; NB],
    weights: [[[[Field; C_OUT]; C_IN]; K]; K],
    bias: [Field; C_OUT],
) -> pub [[[[Field; C_OUT]; W_OUT]; H_OUT]; NB] {
    // Input domain: every input element is a Q32 code in the window.
    for b in 0..NB {
        for r in 0..HEIGHT {
            for c in 0..WIDTH {
                check_inputs(input[b][r][c]);
            }
        }
    }
    for a in 0..K {
        for c in 0..K {
            for ci in 0..C_IN {
                check_inputs(weights[a][c][ci]);
            }
        }
    }
    check_inputs(bias);
    let mut result: [[[[Field; C_OUT]; W_OUT]; H_OUT]; NB] = [[[[0; C_OUT]; W_OUT]; H_OUT]; NB];
    for b in 0..NB {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for co in 0..C_OUT {
                    // np.sum(input[b, i:i + K, j:j + K, :] * weights[:, :, :, co])
                    let mut acc: Field = 0;
                    for a in 0..K {
                        for c in 0..K {
                            for ci in 0..C_IN {
                                acc += input[b][i + a][j + c][ci] * weights[a][c][ci][co];
                            }
                        }
                    }
                    // conv2d_bias: + bias
                    result[b][i][j][co] = rescale(acc) + bias[co];
                }
            }
        }
    }
    result
}
