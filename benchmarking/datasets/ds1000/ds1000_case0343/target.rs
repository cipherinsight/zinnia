//! DS-1000 case 0343, halo2-base baseline.
//! X.T.flat visits columns first; append each input wire in that order.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(30);
    for j in 0..6 {
        for i in 0..5 {
            result.push(x[i * 6 + j]);
        }
    }
    out.ints(&result);
    Ok(())
}
