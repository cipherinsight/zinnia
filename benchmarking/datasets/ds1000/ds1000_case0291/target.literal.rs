//! DS-1000 case 0291: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Shape metadata reads no array elements; constants are omitted from instances.
use crate::prelude::*;

pub fn circuit(_builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let _a = inputs.floats("a", &[2, 5, 5, 2])?;
    // result = a.shape
    let result = [2, 5, 5, 2];
    out.const_ints(&result);
    Ok(())
}
