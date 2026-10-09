//! DS-1000 case 0505: literal translation of kernel.py (LITERAL_BASELINES.md).
//! One standard indexed read per row; any reductions scan the fully built array in index order.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw_a = inputs.floats("a", &[13,13,5])?;
    let raw_b = inputs.ints("b", &[13,13])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &raw_a);
    let b = q.load_ints(ctx, &raw_b);
    // take_along_axis; newaxis and [..., 0] only change the shape.
    let mut arr = Vec::new();
    for i in 0..169 { arr.push(q.index(ctx, &a[i*5..(i+1)*5], b[i])); }
    let mut all = ctx.load_zero();
    for v in &a { all = q.add(ctx, all, *v); }
    let mut total = ctx.load_zero();
    for v in &arr { total = q.add(ctx, total, *v); }
    let result = q.sub(ctx, all, total);
    out.float(result);
    Ok(())
}
