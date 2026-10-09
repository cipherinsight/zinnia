//! DS-1000 case 0304, halo2-base baseline.
//! Kernel: A[3:][::-1], reshaped to (4, 5). Only the 20 selected inputs are read.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("A", &[23])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let selected: Vec<f64> = a[3..].iter().rev().copied().collect();
    let a = q.load_floats(ctx, &selected);
    out.floats(&a);
    Ok(())
}
