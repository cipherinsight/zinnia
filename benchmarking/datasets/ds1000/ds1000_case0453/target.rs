//! DS-1000 case 0453, halo2-base baseline.
//! Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on `Integer[5, 4]`.
//! The squares and their row sum stay integers; the norm is the Q32 square root
//! of that sum, and `/` is NumPy's true division (a Q32 float division). No
//! proof for a zero row (NumPy gives nan) or a row whose sum of squares reaches
//! 2^62 (its norm leaves the Q32 window).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(20);
    for row in x.chunks(4) {
        let sq_sum = q.gate.inner_product(ctx, row.iter().map(|c| Existing(*c)), row.iter().map(|c| Existing(*c)));
        let sq_sum = q.int_to_float(ctx, sq_sum);
        let l2 = q.sqrt(ctx, sq_sum);
        for v in row {
            let v = q.int_to_float(ctx, *v);
            result.push(q.div_pos(ctx, v, l2).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
