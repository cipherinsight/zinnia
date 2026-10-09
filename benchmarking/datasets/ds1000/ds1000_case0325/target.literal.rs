//! DS-1000 case 0325: literal translation of kernel.py (LITERAL_BASELINES.md).
//! deg2rad on integers is exact multiplication by Q32 pi/180; shared trigonometric gadgets.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.int("number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let number = q.load_int(ctx, input);
    let radians = q.mul_int(ctx, number, Constant(fe(quantize(std::f64::consts::PI / 180.0))));
    let radians_b = q.bound(ctx, radians);
    let deg = q.sin(ctx, radians_b);
    let radians = q.int_to_float(ctx, number);
    let radians_b = q.bound(ctx, radians);
    let rad = q.sin(ctx, radians_b);
    let r = q.bound(ctx, rad);
    let d = q.bound(ctx, deg);
    let result = q.gt(ctx, r, d);
    // int(bool) is the same 0/1 integer wire.
    out.int(result);
    Ok(())
}
