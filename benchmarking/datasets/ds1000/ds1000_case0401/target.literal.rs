//! DS-1000 case 0401: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Constant construction at the declared shape; flattened output has 0 values.
use crate::prelude::*;

pub fn circuit(_builder: &mut Builder, _inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    // result = np.array([[], [], []])
    let result: [f64; 0] = [0.0; 0];
    out.const_floats(&result);
    Ok(())
}
