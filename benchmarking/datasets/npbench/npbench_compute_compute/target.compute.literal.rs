//! npbench_compute_compute__compute: literal translation of kernel.py (LITERAL_BASELINES.md).
//! np.clip uses pairwise maximum then minimum; array expressions are separate full passes.
//! Integer arithmetic is exact field arithmetic.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = array_1_0
const COLS: usize = 16; // = array_1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let array_1 = inputs.ints("array_1", &[N, COLS])?;
    let array_2 = inputs.ints("array_2", &[N, COLS])?;
    let (a, b, c) = (inputs.int("a")?, inputs.int("b")?, inputs.int("c")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let array_1 = q.load_ints(ctx, &array_1);
    let array_2 = q.load_ints(ctx, &array_2);
    let a = q.load_int(ctx, a);
    let b = q.load_int(ctx, b);
    let c = q.load_int(ctx, c);

    let clipped: Vec<Cell> = array_1.iter().map(|x| {
        let x = q.bound(ctx, *x);
        let lo = q.bounded_const(ctx, 2);
        let lower = q.max(ctx, x, lo).cell;
        let lower = q.bound(ctx, lower);
        let hi = q.bounded_const(ctx, 10);
        q.min(ctx, lower, hi).cell
    }).collect();
    let term1: Vec<Cell> = clipped.iter().map(|x| q.gate.mul(ctx, *x, a)).collect();
    let term2: Vec<Cell> = array_2.iter().map(|y| q.gate.mul(ctx, *y, b)).collect();
    let sums: Vec<Cell> = term1.iter().zip(&term2).map(|(x,y)| q.add(ctx, *x, *y)).collect();
    let result: Vec<Cell> = sums.iter().map(|x| q.add(ctx, *x, c)).collect();
    out.ints(&result);
    Ok(())
}
