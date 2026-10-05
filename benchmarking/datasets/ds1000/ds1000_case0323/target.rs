//! DS-1000 case 0323, halo2-base baseline.
//! Kernel: `np.sin(np.deg2rad(degree))` for an `Integer` degree.
//! `deg2rad` of an integer is the integer times the Q32 constant pi/180 (exact,
//! no rescale).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let degree = inputs.int("degree")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let degree = q.load_int(ctx, degree);
    let rad = q.mul_int(ctx, degree, Constant(fe(quantize(std::f64::consts::PI / 180.0))));
    let rad = q.bound(ctx, rad);
    let s = q.sin(ctx, rad);
    out.float(s);
    Ok(())
}
