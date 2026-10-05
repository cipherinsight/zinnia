//! DS-1000 case 0454, halo2-base baseline.
//! Kernel: `X / np.abs(X).max(axis=1).reshape(-1, 1)` on `Integer[5, 4]`.
//! The row norm max|x| is `q.max_of` over the 8 values x and -x of the row.
//! `/` is NumPy's true division, so each output is the Q32 quotient
//! floor(x * 2^32 / linf) of two integers; `q.div_pos` on the raw integers
//! computes exactly that (it scales its numerator by 2^32). A zero row
//! (linf = 0, NumPy gives nan) has no proof.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(20);
    for row in x.chunks(4) {
        let mut signed = Vec::with_capacity(8);
        for v in row {
            signed.push(*v);
            signed.push(q.neg(ctx, *v));
        }
        let linf = q.max_of(ctx, &signed);
        let linf = q.bound(ctx, linf);
        for v in row {
            result.push(q.div_pos(ctx, *v, linf).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
