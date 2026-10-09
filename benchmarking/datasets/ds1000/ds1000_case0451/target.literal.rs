//! DS-1000 case 0451: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Constant construction at the declared shape; flattened output has 4000 values.
use crate::prelude::*;

pub fn circuit(_builder: &mut Builder, _inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    // arr = np.zeros((20,10,10,2))
    let result: [f64; 4000] = [0.0; 4000];
    out.const_floats(&result);
    Ok(())
}
