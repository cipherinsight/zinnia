// pythran_multi_export__a: literal translation of kernel.py (LITERAL_BASELINES.md).
// Identity return; only input-domain checking adds constraints.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let i = inputs.float("i")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let i = q.load_float(ctx, i);
    out.float(i);
    Ok(())
}
