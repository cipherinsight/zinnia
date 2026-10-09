//! DS-1000 case 0342, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel: `result = []; for value in X.flat: result.append(value)` on
//! `Integer[5, 6]`. `X.flat` visits the 30 elements in row-major order, which
//! is already the array's storage order, so the loop over `X.flat` is a
//! straight copy: each loaded input cell becomes the output at the same
//! position. There is no arithmetic to translate.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // for value in X.flat: result.append(value)
    let x = q.load_ints(ctx, &x);
    let mut result = Vec::new();
    for value in x { result.push(value); }
    out.ints(&result);
    Ok(())
}
