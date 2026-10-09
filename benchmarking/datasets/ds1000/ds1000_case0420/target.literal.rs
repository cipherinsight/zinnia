//! DS-1000 case 0420: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Both where branches computed eagerly; powers computed independently; integer literals multiply Q32 exactly.
//! L4: select zero for the polynomial operand when either clamp applies;
//! otherwise unused powers can overflow Q32 for valid inputs (e.g. x=10000).
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let xv = inputs.float("x")?;
    let lv = inputs.int("x_min")?;
    let hv = inputs.int("x_max")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_float(ctx, xv);
    let lo = q.load_int(ctx, lv);
    let hi = q.load_int(ctx, hv);
    let lo_cmp = q.int_to_float(ctx, lo);
    let xb = q.bound(ctx, x);
    let lb = q.bound(ctx, lo_cmp);
    let below = q.lt(ctx, xb, lb);
    let hi_cmp = q.int_to_float(ctx, hi);
    let xb = q.bound(ctx, x);
    let hb = q.bound(ctx, hi_cmp);
    let above = q.gt(ctx, xb, hb);
    let inactive = q.gate.or(ctx, below, above);
    let t = q.gate.select(ctx, Constant(fe(0)), x, inactive);
    let square = q.pow_const(ctx, t, 2);
    let first = q.mul_int(ctx, square, Constant(fe(3)));
    let cube = q.pow_const(ctx, t, 3);
    let second = q.mul_int(ctx, cube, Constant(fe(2)));
    let smooth = q.sub(ctx, first, second);
    let hi_result = q.int_to_float(ctx, hi);
    let inner = q.gate.select(ctx, hi_result, smooth, above);
    let lo_result = q.int_to_float(ctx, lo);
    let result = q.gate.select(ctx, lo_result, inner, below);
    out.float(result);
    Ok(())
}
