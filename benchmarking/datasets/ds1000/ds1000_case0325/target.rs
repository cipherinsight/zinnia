//! DS-1000 case 0325, halo2-base baseline.
//! Kernel: `int(np.sin(number) > np.sin(np.deg2rad(number)))` for an `Integer`
//! number (does `number` read as radians give a larger sine than as degrees).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let number = inputs.int("number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let number = q.load_int(ctx, number);
    let as_deg = q.mul_int(ctx, number, Constant(fe(quantize(std::f64::consts::PI / 180.0))));
    let as_deg = q.bound(ctx, as_deg);
    let as_rad = q.int_to_float(ctx, number);
    let as_rad = q.bound(ctx, as_rad);
    let deg = q.sin(ctx, as_deg);
    let rad = q.sin(ctx, as_rad);
    let (deg, rad) = (q.bound(ctx, deg), q.bound(ctx, rad));
    let result = q.gt(ctx, rad, deg);
    out.int(result);
    Ok(())
}
