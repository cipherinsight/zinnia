//! npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias (NPBench
//! deep_learning/conv2d_bias), halo2-base baseline.
//! Kernel: `conv2d(input, weights) + bias` on `input: Float[NB, HEIGHT,
//! WIDTH, C_IN]`, `weights: Float[K, K, C_IN, C_OUT]`, `bias: Float[C_OUT]`:
//! output `[b, i, j, co]` is `np.sum(input[b, i:i + K, j:j + K, :] *
//! weights[:, :, :, co]) + bias[co]` for `i < HEIGHT - K + 1`,
//! `j < WIDTH - K + 1`.
//! Each sum of products is one reduction, rescaled once (`q.dot`,
//! `(x * y).sum()` in CONVENTIONS section 2). `conv2d` stores it in a
//! float32 array: computed in Q32 like any float (NPB_BASELINES.md section
//! 5.4; no discrete outcome depends on it). Adding the bias is exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NB: usize = 4; // = input_0
const HEIGHT: usize = 16; // = input_1
const WIDTH: usize = 16; // = input_2
const C_IN: usize = 2; // = input_3
const K: usize = 2; // = weights_0
const C_OUT: usize = 8; // = weights_3
const H_OUT: usize = HEIGHT - K + 1;
const W_OUT: usize = WIDTH - K + 1;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.floats("input", &[NB, HEIGHT, WIDTH, C_IN])?;
    let weights = inputs.floats("weights", &[K, K, C_IN, C_OUT])?;
    let bias = inputs.floats("bias", &[C_OUT])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let input = q.load_floats(ctx, &input);
    let weights = q.load_floats(ctx, &weights);
    let bias = q.load_floats(ctx, &bias);
    let pixel = |b: usize, r: usize, c: usize, ci: usize| ((b * HEIGHT + r) * WIDTH + c) * C_IN + ci;
    // The filter of each output channel, in the window's order (a, b', ci).
    let filters: Vec<Vec<Cell>> =
        (0..C_OUT).map(|co| (0..K * K * C_IN).map(|t| weights[t * C_OUT + co]).collect()).collect();
    let mut result: Vec<Cell> = Vec::with_capacity(NB * H_OUT * W_OUT * C_OUT);
    for b in 0..NB {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                let mut window: Vec<Cell> = Vec::with_capacity(K * K * C_IN);
                for a in 0..K {
                    for c in 0..K {
                        window.extend((0..C_IN).map(|ci| input[pixel(b, i + a, j + c, ci)]));
                    }
                }
                for (filter, bias) in filters.iter().zip(&bias) {
                    let sum = q.dot(ctx, &window, filter);
                    result.push(q.add(ctx, sum.cell, *bias));
                }
            }
        }
    }
    out.floats(&result);
    Ok(())
}
