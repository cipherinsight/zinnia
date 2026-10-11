//! DS-1000 case 0453, halo2-base baseline.
//! Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on `Integer[5, 4]`.
//! The squares and their row sum stay integers; the norm is the Q32 square root
//! of that sum, and `/` is NumPy's true division (a Q32 float division). No
//! proof for a zero row (NumPy gives nan), or sum(X*X) >= 2^31, matching
//! the compiled Halo2 circuit's signed64 check of the Q32 radicand.
//! The int64 input checks make each four-square sum an exact integer at most
//! 2^128 (and its Q32 scaling at most 2^160), so field wrap cannot evade the
//! 31-bit row check. It implies all former 32-bit input bounds; no separate
//! per-element bounds or nonzero check are needed (division rejects zero).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(20);
    for row in x.chunks(4) {
        let sq_sum = q.gate.inner_product(ctx, row.iter().map(|c| Existing(*c)), row.iter().map(|c| Existing(*c)));
        q.range.range_check(ctx, sq_sum, 31);
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
