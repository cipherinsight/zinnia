//! DS-1000 case 0324, halo2-base baseline.
//! Kernel: `np.cos(np.deg2rad(degree))` for an `Integer` degree.
//! `deg2rad` of an integer is the integer times the Q32 constant pi/180 (exact,
//! no rescale); `cos(x) = sin(x + pi/2)`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let degree = inputs.int("degree")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let degree = q.load_int(ctx, degree);
    let rad = q.mul_int(ctx, degree, Constant(fe(quantize(std::f64::consts::PI / 180.0))));
    let c = q.cos(ctx, rad);
    out.float(c);
    Ok(())
}
