// pythran_lap__laplacian: literal translation of kernel.py (LITERAL_BASELINES.md).
// Nine slice statements are evaluated separately, retaining lap[0]=4/3*var[1]. Fresh division gadget per output.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = var_0

const NEAR: f64 = 4.0 / 3.0;
const CENTER: f64 = -5.0 / 2.0;
const FAR: f64 = -1.0 / 12.0;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let var = inputs.floats("var", &[N])?;
    let dh2 = inputs.float("dh2")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let var = q.load_floats(ctx, &var);
    let dh2 = q.load_float(ctx, dh2);
    let zero = q.float_const(ctx, 0.0);
    let mut lap = vec![zero; N];
    let products: Vec<Cell> = (1..N)
        .map(|i| q.mul(ctx, Constant(fe(quantize(NEAR))), var[i - 1]).cell)
        .collect();
    for i in 1..N {
        lap[i] = products[i - (1)];
    }
    let products: Vec<Cell> = (0..1)
        .map(|_i| q.mul(ctx, Constant(fe(quantize(NEAR))), var[1]).cell)
        .collect();
    for i in 0..1 {
        lap[i] = products[i - (0)];
    }
    let products: Vec<Cell> = (0..N - 1)
        .map(|i| q.mul(ctx, Constant(fe(quantize(NEAR))), var[i + 1]).cell)
        .collect();
    for i in 0..N - 1 {
        lap[i] = q.add(ctx, lap[i], products[i - (0)]);
    }
    let products: Vec<Cell> = (N - 1..N)
        .map(|_i| q.mul(ctx, Constant(fe(quantize(NEAR))), var[0]).cell)
        .collect();
    for i in N - 1..N {
        lap[i] = q.add(ctx, lap[i], products[i - (N - 1)]);
    }
    let products: Vec<Cell> = (0..N)
        .map(|i| q.mul(ctx, Constant(fe(quantize(CENTER))), var[i]).cell)
        .collect();
    for i in 0..N {
        lap[i] = q.add(ctx, lap[i], products[i - (0)]);
    }
    let products: Vec<Cell> = (2..N)
        .map(|i| q.mul(ctx, Constant(fe(quantize(FAR))), var[i - 2]).cell)
        .collect();
    for i in 2..N {
        lap[i] = q.add(ctx, lap[i], products[i - (2)]);
    }
    let products: Vec<Cell> = (0..2)
        .map(|i| q.mul(ctx, Constant(fe(quantize(FAR))), var[N - 2 + i]).cell)
        .collect();
    for i in 0..2 {
        lap[i] = q.add(ctx, lap[i], products[i - (0)]);
    }
    let products: Vec<Cell> = (0..N - 2)
        .map(|i| q.mul(ctx, Constant(fe(quantize(FAR))), var[i + 2]).cell)
        .collect();
    for i in 0..N - 2 {
        lap[i] = q.add(ctx, lap[i], products[i - (0)]);
    }
    let products: Vec<Cell> = (N - 2..N)
        .map(|i| {
            q.mul(ctx, Constant(fe(quantize(FAR))), var[i - (N - 2)])
                .cell
        })
        .collect();
    for i in N - 2..N {
        lap[i] = q.add(ctx, lap[i], products[i - (N - 2)]);
    }
    let result: Vec<Cell> = lap
        .iter()
        .map(|v| q.canonical_div_nonzero(ctx, *v, dh2).cell)
        .collect();
    out.floats(&result);
    Ok(())
}
