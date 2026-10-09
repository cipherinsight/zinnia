// npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias (NPBench
// deep_learning/conv2d_bias), Noir baseline.
// Kernel: conv2d(input, weights) + bias on
// input: Float[NB, HEIGHT, WIDTH, C_IN], weights: Float[K, K, C_IN, C_OUT],
// bias: Float[C_OUT]: output [b, i, j, co] is
// np.sum(input[b, i:i + K, j:j + K, :] * weights[:, :, :, co]) + bias[co] for
// i < HEIGHT - K + 1, j < WIDTH - K + 1.
// Each sum of products is one reduction, rescaled once ((x * y).sum() in
// CONVENTIONS section 2). conv2d stores it in a float32 array: computed in Q32
// like any float (NPB_BASELINES.md section 5.4; no discrete outcome depends on
// it). Adding the bias is exact.
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
                    let mut acc: Field = 0;
                    for a in 0..K {
                        for c in 0..K {
                            for ci in 0..C_IN {
                                acc += input[b][i + a][j + c][ci] * weights[a][c][ci][co];
                            }
                        }
                    }
                    result[b][i][j][co] = rescale(acc) + bias[co];
                }
            }
        }
    }
    result
}
