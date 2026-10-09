//! npbench_deep_learning_conv2d_bias_conv2d__conv2d_bias.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Broadcast products are materialized and rounded before axis-order sums.
//! The conv2d helper completes before the broadcast bias addition.

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
    let convolution = conv2d(&q, ctx, &input, &weights);
    let result: Vec<Cell> = (0..NB*H_OUT*W_OUT*C_OUT)
        .map(|i| q.add(ctx, convolution[i], bias[i%C_OUT])).collect();
    out.floats(&result);
    Ok(())
}

fn conv2d(q: &Q32, ctx: &mut Ctx, input: &[Cell], weights: &[Cell]) -> Vec<Cell> {
    let zero = q.float_const(ctx, 0.0);
    let mut output = vec![zero; NB*H_OUT*W_OUT*C_OUT];
    for i in 0..H_OUT {
        for j in 0..W_OUT {
            let mut products = Vec::with_capacity(NB*K*K*C_IN*C_OUT);
            for b in 0..NB { for a in 0..K { for c in 0..K { for ci in 0..C_IN { for co in 0..C_OUT {
                products.push(q.mul(ctx, input[((b*HEIGHT+i+a)*WIDTH+j+c)*C_IN+ci],
                    weights[((a*K+c)*C_IN+ci)*C_OUT+co]).cell);
            } } } } }
            for b in 0..NB { for co in 0..C_OUT {
                let mut acc = zero;
                for a in 0..K { for c in 0..K { for ci in 0..C_IN {
                    acc = q.add(ctx, acc, products[(((b*K+a)*K+c)*C_IN+ci)*C_OUT+co]);
                } } }
                output[((b*H_OUT+i)*W_OUT+j)*C_OUT+co] = acc;
            } }
        }
    }
    output
}
