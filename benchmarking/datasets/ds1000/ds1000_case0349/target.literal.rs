//! DS-1000 case 0349: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Shape metadata reads no array elements; constants are omitted from instances.
use crate::prelude::*;

pub fn circuit(_builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let _a = inputs.floats("a", &[8, 6])?;
    // result = a.shape[1]
    let result = [6];
    out.const_ints(&result);
    Ok(())
}
