//! DS-1000 case 0323: literal translation of kernel.py (LITERAL_BASELINES.md).
//! deg2rad on integers is exact multiplication by Q32 pi/180; shared trigonometric gadgets.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.int("degree")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let number = q.load_int(ctx, input);
    let radians = q.mul_int(ctx, number, Constant(fe(quantize(std::f64::consts::PI / 180.0))));
    let radians_b = q.bound(ctx, radians);
    let deg = q.sin(ctx, radians_b);
    out.float(deg);
    Ok(())
}
