// npbench_deep_learning_resnet_resnet__resnet_basicblock.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Convolution materializes rounded elementwise products, then sums in axis order.
// std: mean, centered rounded squares, mean, sqrt; denominator is sqrt(std+eps).
// Q32 throughout: float32 array storage adds no rounding step. All indices are static.
use crate::prelude::*;
// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 4; // = input_0
const H: usize = 7; // = input_1
const W: usize = 7; // = input_2
const CI: usize = 16; // = input_3
const CM: usize = 4; // = conv1_3
const K1: usize = 1; // = conv1_0
const K2: usize = 3; // = conv2_0
const K3: usize = 1; // = conv3_0
const CO: usize = 16; // = conv3_3
const PH: usize = H + 2;
const PW: usize = W + 2;
const H1: usize = H - K1 + 1;
const W1: usize = W - K1 + 1;
const H2: usize = PH - K2 + 1;
const W2: usize = PW - K2 + 1;
const H3: usize = H2 - K3 + 1;
const W3: usize = W2 - K3 + 1;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let input = q.load_floats(ctx, &inputs.floats("input", &[N, H, W, CI])?);
    let conv1 = q.load_floats(ctx, &inputs.floats("conv1", &[K1, K1, CI, CM])?);
    let conv2 = q.load_floats(ctx, &inputs.floats("conv2", &[K2, K2, CM, CM])?);
    let conv3 = q.load_floats(ctx, &inputs.floats("conv3", &[K3, K3, CM, CO])?);
    let zero = q.float_const(ctx, 0.0);
    let mut padded = vec![zero; N * PH * PW * CM];
    let first = conv2d(&q, ctx, &input, &conv1, H, W, CI, CM, K1);
    for b in 0..N {
        for i in 0..H1 {
            for j in 0..W1 {
                for c in 0..CM {
                    padded[((b * PH + i + 1) * PW + j + 1) * CM + c] =
                        first[((b * H1 + i) * W1 + j) * CM + c];
                }
            }
        }
    }
    let x = batchnorm2d(&q, ctx, &padded);
    let x = relu(&q, ctx, &x);
    let x = conv2d(&q, ctx, &x, &conv2, PH, PW, CM, CM, K2);
    let x = batchnorm2d(&q, ctx, &x);
    let x = relu(&q, ctx, &x);
    let x = conv2d(&q, ctx, &x, &conv3, H2, W2, CM, CO, K3);
    let x = batchnorm2d(&q, ctx, &x);
    let sum: Vec<_> = x
        .iter()
        .zip(&input)
        .map(|(&a, &b)| q.add(ctx, a, b))
        .collect();
    out.floats(&relu(&q, ctx, &sum));
    Ok(())
}
fn relu(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<Cell> {
    x.iter()
        .map(|&v| {
            let v = q.bound(ctx, v);
            let z = q.bounded_const(ctx, 0);
            q.max(ctx, v, z).cell
        })
        .collect()
}
fn mean(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<Cell> {
    let s = x.len() / N;
    (0..s)
        .map(|i| {
            let mut sum = q.float_const(ctx, 0.0);
            for b in 0..N {
                sum = q.add(ctx, sum, x[b * s + i]);
            }
            q.div_const(ctx, sum, N as f64).cell
        })
        .collect()
}
fn batchnorm2d(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<Cell> {
    let s = x.len() / N;
    let avg = mean(q, ctx, x);
    let std_avg = mean(q, ctx, x);
    let deviations: Vec<_> = (0..x.len())
        .map(|i| q.sub(ctx, x[i], std_avg[i % s]))
        .collect();
    let squares: Vec<_> = deviations.iter().map(|&d| q.mul(ctx, d, d).cell).collect();
    let variance = mean(q, ctx, &squares);
    let std: Vec<_> = variance.iter().map(|&v| q.sqrt(ctx, v).cell).collect();
    let numerator: Vec<_> = (0..x.len()).map(|i| q.sub(ctx, x[i], avg[i % s])).collect();
    let eps = q.float_const(ctx, 1e-5);
    let shifted: Vec<_> = std.iter().map(|&v| q.add(ctx, v, eps)).collect();
    let denominator: Vec<_> = shifted.iter().map(|&v| q.sqrt(ctx, v)).collect();
    (0..x.len())
        .map(|i| {
            let d = q.bound(ctx, denominator[i % s].cell);
            q.div(ctx, numerator[i], d).cell
        })
        .collect()
}
fn conv2d(
    q: &Q32,
    ctx: &mut Ctx,
    input: &[Cell],
    weights: &[Cell],
    h: usize,
    w: usize,
    ci: usize,
    co: usize,
    k: usize,
) -> Vec<Cell> {
    let oh = h - k + 1;
    let ow = w - k + 1;
    let zero = q.float_const(ctx, 0.0);
    let mut output = vec![zero; N * oh * ow * co];
    for i in 0..oh {
        for j in 0..ow {
            let mut products = Vec::with_capacity(N * k * k * ci * co);
            for b in 0..N {
                for a in 0..k {
                    for d in 0..k {
                        for c in 0..ci {
                            for o in 0..co {
                                products.push(
                                    q.mul(
                                        ctx,
                                        input[((b * h + i + a) * w + j + d) * ci + c],
                                        weights[((a * k + d) * ci + c) * co + o],
                                    )
                                    .cell,
                                );
                            }
                        }
                    }
                }
            }
            for b in 0..N {
                for o in 0..co {
                    let mut sum = zero;
                    for a in 0..k {
                        for d in 0..k {
                            for c in 0..ci {
                                sum = q.add(
                                    ctx,
                                    sum,
                                    products[(((b * k + a) * k + d) * ci + c) * co + o],
                                );
                            }
                        }
                    }
                    output[((b * oh + i) * ow + j) * co + o] = sum;
                }
            }
        }
    }
    output
}
