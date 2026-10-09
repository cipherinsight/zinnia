// pythran_clip__clip: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each abs occurrence is recomputed; complex magnitude/division retain the optimized Q32 arithmetic family. Inactive zero division is totalized and validity gated.
use crate::prelude::*;
const N: usize = 2; // = z_0
fn magnitude(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Cell {
    let rr = q.mul(ctx, x[0], x[0]).cell;
    let ii = q.mul(ctx, x[1], x[1]).cell;
    let sum = q.add(ctx, rr, ii);
    q.sqrt(ctx, sum).cell
}
fn limit1(q: &Q32, ctx: &mut Ctx, x: &[Cell], outer: Cell) -> [Cell; 2] {
    let mag = magnitude(q, ctx, x);
    let mb = q.bound(ctx, mag);
    let eps = q.bounded_const(ctx, quantize(1e-6));
    let tiny = q.lt(ctx, mb, eps);
    let mag = magnitude(q, ctx, x);
    // Complex/real division: retain this case's rounded numerator and denominator contract.
    let den = q.mul(ctx, mag, mag).cell;
    let not_tiny = q.gate.not(ctx, tiny);
    let active = q.gate.and(ctx, outer, not_tiny);
    let zero = q.eq(ctx, den, Constant(Fr::ZERO));
    let invalid = q.gate.and(ctx, active, zero);
    q.gate.assert_is_const(ctx, &invalid, &Fr::ZERO);
    let mut unit = [q.float_const(ctx, 0.0); 2];
    for part in 0..2 {
        let num = q.mul(ctx, x[part], mag).cell;
        let quotient = q.canonical_div_or_zero(ctx, num, den).cell;
        unit[part] = q.gate.select(ctx, Constant(Fr::ZERO), quotient, tiny);
    }
    unit
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let z = inputs.complexes("z", &[N])?;
    let max = inputs.float("_max")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let z = q.load_floats(ctx, &z);
    let max = q.load_float(ctx, max);
    let mut result = Vec::with_capacity(2 * N);
    for i in 0..N {
        let x = &z[2 * i..2 * i + 2];
        let mag = magnitude(&q, ctx, x);
        let mb = q.bound(ctx, mag);
        let xb = q.bound(ctx, max);
        let over = q.gt(ctx, mb, xb);
        let unit = limit1(&q, ctx, x, over);
        let limited: Vec<Cell> = unit.iter().map(|v| q.mul(ctx, *v, max).cell).collect();
        for part in 0..2 {
            result.push(q.gate.select(ctx, limited[part], x[part], over));
        }
    }
    out.floats(&result);
    Ok(())
}
