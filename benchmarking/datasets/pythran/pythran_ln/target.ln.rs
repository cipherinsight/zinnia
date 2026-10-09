//! pythran_ln__ln (Pythran ln), halo2-base baseline.
//! Kernel: `Y[:] = d - d ** 2 / 2 + d ** 3 / 3 - ... + d ** 9 / 9` with
//! `d = X - 1` on X, Y: Float[N]; returns `Y`. `Y` is overwritten before it
//! is read, so it is not loaded.
//! Each power is `q.pow_const`'s square-and-multiply product chain; the chains
//! of the eight powers share their squares and `d ** 3`, so the eight powers
//! cost eight rounded products. `v / n` by the integer constant `n` is
//! `floor(v * 2^32 / (n * 2^32)) = floor(v / n)`: `q.floor_div_const(v, n)`,
//! whose remainder check is against `n` itself.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = X_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("X", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    let result: Vec<Cell> = x
        .iter()
        .map(|v| {
            let d = q.sub(ctx, *v, Constant(fe(SCALE)));
            let powers = powers_to_9(&q, ctx, d);
            let mut y = d;
            for (n, power) in powers.iter().enumerate().skip(2) {
                let (term, _) = q.floor_div_const(ctx, *power, n as u64);
                y = if n % 2 == 0 { q.sub(ctx, y, term.cell) } else { q.add(ctx, y, term.cell) };
            }
            y
        })
        .collect();
    out.floats(&result);
    Ok(())
}

/// `[_, d, d ** 2, ..., d ** 9]`, each power with the products and rounding
/// of `q.pow_const(d, n)` (right-to-left square and multiply), shared.
fn powers_to_9(q: &Q32, ctx: &mut Ctx, d: Cell) -> [Cell; 10] {
    let mut mul = |a: Cell, b: Cell| q.mul(ctx, a, b).cell;
    let d2 = mul(d, d);
    let d4 = mul(d2, d2);
    let d8 = mul(d4, d4);
    let d3 = mul(d, d2);
    let d5 = mul(d, d4);
    let d6 = mul(d2, d4);
    let d7 = mul(d3, d4);
    let d9 = mul(d, d8);
    [d, d, d2, d3, d4, d5, d6, d7, d8, d9]
}
