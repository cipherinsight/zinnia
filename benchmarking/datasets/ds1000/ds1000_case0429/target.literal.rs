//! DS-1000 case 0429: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each mean/std occurrence is independent; std uses mean, full deviations, rounded squares, mean and sqrt.
use crate::prelude::*;

const N: usize = 30;
fn mean(q: &Q32, ctx: &mut Ctx, a: &[Cell]) -> Cell {
    let mut total = q.int_const(ctx, 0);
    for x in a { total = q.add(ctx, total, *x); }
    q.floor_div_const(ctx, total, N as u64).0.cell
}
fn std(q: &Q32, ctx: &mut Ctx, a: &[Cell]) -> Cell {
    let avg = mean(q, ctx, a);
    let dev: Vec<Cell> = a.iter().map(|x| q.sub(ctx, *x, avg)).collect();
    let squares: Vec<Cell> = dev.iter().map(|x| q.mul(ctx, *x, *x).cell).collect();
    let mut total = q.int_const(ctx, 0);
    for x in squares { total = q.add(ctx, total, x); }
    let variance = q.floor_div_const(ctx, total, N as u64).0.cell;
    q.sqrt(ctx, variance).cell
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &av);
    let m0 = mean(&q, ctx, &a);
    let s0 = std(&q, ctx, &a);
    let spread0 = q.mul_int(ctx, s0, Constant(fe(2)));
    let low = q.sub(ctx, m0, spread0);
    let m1 = mean(&q, ctx, &a);
    let s1 = std(&q, ctx, &a);
    let spread1 = q.mul_int(ctx, s1, Constant(fe(2)));
    let high = q.add(ctx, m1, spread1);
    out.floats(&[low, high]);
    Ok(())
}
