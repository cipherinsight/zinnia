//! DS-1000 case 0330, halo2-base baseline.
//! Kernel: `a ** power` with the constant `power = 6` on `Float[8, 6]`.
//! Square-and-multiply (`q.pow_const`): `x^2`, `x^4 = x^2 * x^2`, `x^2 * x^4`,
//! three rescaled products per element.

use crate::prelude::*;

const POWER: u32 = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let powered: Vec<Cell> = a.iter().map(|x| q.pow_const(ctx, *x, POWER)).collect();
    out.floats(&powered);
    Ok(())
}
