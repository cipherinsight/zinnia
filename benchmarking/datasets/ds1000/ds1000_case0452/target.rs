//! DS-1000 case 0452, halo2-base baseline.
//! Kernel: `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on `Integer[5, 4]`.
//! `/` is NumPy's true division, so each output is the Q32 quotient
//! floor(x * 2^32 / l1) of two integers; `q.div_pos` on the raw integers
//! computes exactly that (it scales its numerator by 2^32). |x| is
//! `q.max_of([x, -x])` (cheaper than `q.abs`, which needs a window proof and a
//! comparison). A zero row (l1 = 0, NumPy gives nan) has no proof.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(20);
    for row in x.chunks(4) {
        let abs: Vec<Cell> = row
            .iter()
            .map(|v| {
                let minus_v = q.neg(ctx, *v);
                q.max_of(ctx, &[*v, minus_v])
            })
            .collect();
        let l1 = q.sum(ctx, abs);
        let l1 = q.bound(ctx, l1);
        for v in row {
            result.push(q.div_pos(ctx, *v, l1).cell);
        }
    }
    out.floats(&result);
    Ok(())
}
