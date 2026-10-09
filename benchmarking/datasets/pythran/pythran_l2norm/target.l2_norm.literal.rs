//! pythran_l2norm__l2_norm: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize abs, squares, sums, then roots; retain the optimized root-domain guard.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = x_0
const COLS: usize = 16; // = x_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("x", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &raw);
    let abs: Vec<Cell> = x.iter().map(|v| q.abs_int64(ctx, *v)).collect();
    let squares: Vec<Cell> = abs.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let sums: Vec<Cell> = squares
        .chunks(COLS)
        .map(|row| {
            let mut acc = q.float_const(ctx, 0.0);
            for v in row {
                acc = q.add(ctx, acc, *v);
            }
            acc
        })
        .collect();
    let roots: Vec<Cell> = sums
        .iter()
        .map(|v| {
            q.range.range_check(ctx, *v, 63);
            q.sqrt(ctx, *v).cell
        })
        .collect();
    out.floats(&roots);
    Ok(())
}
