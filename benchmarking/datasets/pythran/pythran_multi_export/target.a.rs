//! pythran_multi_export__a (Pythran multi_export), halo2-base baseline.
//! Kernel: `return i` on a scalar `i: Float`. Data movement only: the output
//! is the input, loaded with its domain check (NPB_BASELINES.md, section 5,
//! rule 6).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let i = inputs.float("i")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let i = q.load_float(ctx, i);
    out.float(i);
    Ok(())
}
