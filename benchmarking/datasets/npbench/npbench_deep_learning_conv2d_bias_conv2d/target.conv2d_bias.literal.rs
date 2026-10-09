//! npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias, halo2-base baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! `conv2d_bias` calls `conv2d(input, weights) + bias`; `conv2d`'s explicit
//! Python loops are `for i in range(H_out): for j in range(W_out):`, with one
//! array statement per (i, j):
//!   output[:, i, j, :] = np.sum(
//!       input[:, i:i + K, j:j + K, :, newaxis] * weights[newaxis, :, :, :],
//!       axis=(1, 2, 3))
//! The batch axis `b` and output-channel axis `co` are not explicit loop
//! variables; they come from broadcasting the two operands against each
//! other, so a literal translation loops over them in the array's own axis
//! order (L5, "element-wise ops, broadcasting"). For each (b, i, j, co), the
//! statement is `np.sum(x * y)` over the window and filter: one reduction,
//! rescaled once (`q.dot`, CONVENTIONS section 2). Slicing out the window and
//! each output channel's filter is data selection, not arithmetic (no gate),
//! so collecting them before the dot call is not hoisting computation (L3
//! forbids hoisting arithmetic work, not re-pointing at existing cells).
//! `conv2d_bias`'s `+ bias` is exact, elementwise over `co` (L5 broadcasting).
//! This is the direct translation already: there is no redundant arithmetic
//! in this kernel to eliminate, so it is structurally identical to the
//! optimized baseline.

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
    // weights[newaxis, :, :, :], one filter per output channel co, in the
    // window's own (a, c, ci) order.
    let filters: Vec<Vec<Cell>> =
        (0..C_OUT).map(|co| (0..K * K * C_IN).map(|t| weights[t * C_OUT + co]).collect()).collect();
    let mut result: Vec<Cell> = Vec::with_capacity(NB * H_OUT * W_OUT * C_OUT);
    for b in 0..NB {
        for i in 0..H_OUT {
            for j in 0..W_OUT {
                // input[b, i:i + K, j:j + K, :]
                let mut window: Vec<Cell> = Vec::with_capacity(K * K * C_IN);
                for a in 0..K {
                    for c in 0..K {
                        window.extend((0..C_IN).map(|ci| input[pixel(b, i + a, j + c, ci)]));
                    }
                }
                for (filter, bias) in filters.iter().zip(&bias) {
                    // np.sum(window * filter)
                    let sum = q.dot(ctx, &window, filter);
                    // conv2d_bias: + bias
                    result.push(q.add(ctx, sum.cell, *bias));
                }
            }
        }
    }
    out.floats(&result);
    Ok(())
}
