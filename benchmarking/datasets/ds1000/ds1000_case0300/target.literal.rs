//! DS-1000 case 0300: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Percentile: stable bubble sort (190 swaps), exact integer rank split,
//! Q32 fractional weight, then the linear interpolation s_lo + weight * (s_hi-s_lo).
//! p is checked in [0,100]; endpoint hi is clamped to the last index.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[20])?;
    let p = inputs.int("p")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut s = q.load_floats(ctx, &a);
    let p = q.load_int(ctx, p);
    q.range.check_less_than_safe(ctx, p, 101);
    for pass in 0..19 {
        for j in 0..19-pass {
            let a = q.bound(ctx, s[j]);
            let b = q.bound(ctx, s[j+1]);
            let swap = q.gt(ctx, a, b);
            let left = q.gate.select(ctx, s[j+1], s[j], swap);
            let right = q.gate.select(ctx, s[j], s[j+1], swap);
            s[j] = left;
            s[j+1] = right;
        }
    }
    let rank_num = q.gate.mul(ctx, p, Constant(fe(19)));
    let (lo, rem) = q.floor_div_const(ctx, rank_num, 100);
    let next = q.add(ctx, lo.cell, Constant(fe(1)));
    let next_b = q.bound(ctx, next);
    let last = q.int_const(ctx, 19);
    let last_b = q.bound(ctx, last);
    let hi = q.min(ctx, next_b, last_b).cell;
    let s_lo = q.index(ctx, &s, lo.cell);
    let s_hi = q.index(ctx, &s, hi);
    let rem_float = q.int_to_float(ctx, rem);
    let weight = q.div_const(ctx, rem_float, 100.0).cell;
    let diff = q.sub(ctx, s_hi, s_lo);
    let contribution = q.mul(ctx, weight, diff).cell;
    let result = q.add(ctx, s_lo, contribution);
    out.float(result);
    Ok(())
}
