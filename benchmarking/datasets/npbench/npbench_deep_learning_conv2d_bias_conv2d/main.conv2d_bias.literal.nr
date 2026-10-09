// npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Broadcast products are materialized and rounded before axis-order sums.
// The conv2d helper completes before the broadcast bias addition.
// @outputs: float[NB * H_OUT * W_OUT * C_OUT]

use noir_hw::{check_inputs, mul};

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
    let convolution = conv2d(input, weights);
    let mut result = [[[[0; C_OUT]; W_OUT]; H_OUT]; NB];
    for b in 0..NB {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                for co in 0..C_OUT {
                    result[b][i][j][co] = convolution[b][i][j][co] + bias[co];
                }
            }
        }
    }
    result
}

fn conv2d(
    input: [[[[Field; C_IN]; WIDTH]; HEIGHT]; NB],
    weights: [[[[Field; C_OUT]; C_IN]; K]; K],
) -> [[[[Field; C_OUT]; W_OUT]; H_OUT]; NB] {
    let mut output = [[[[0; C_OUT]; W_OUT]; H_OUT]; NB];
    for i in 0..H_OUT {
        for j in 0..W_OUT {
            let mut products = [[[[[0; C_OUT]; C_IN]; K]; K]; NB];
            for b in 0..NB {
                for a in 0..K {
                    for c in 0..K {
                        for ci in 0..C_IN {
                            for co in 0..C_OUT {
                                products[b][a][c][ci][co] =
                                    mul(input[b][i + a][j + c][ci], weights[a][c][ci][co]);
                            }
                        }
                    }
                }
            }
            for b in 0..NB {
                for co in 0..C_OUT {
                    let mut acc: Field = 0;
                    for a in 0..K {
                        for c in 0..K {
                            for ci in 0..C_IN {
                                acc += products[b][a][c][ci][co];
                            }
                        }
                    }
                    output[b][i][j][co] = acc;
                }
            }
        }
    }
    output
}
