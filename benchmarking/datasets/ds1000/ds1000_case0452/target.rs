//! DS-1000 case 0452, halo2-base baseline.
//! Kernel: `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on `Integer[5, 4]`.
//! `/` is NumPy's true division, so each output is the Q32 quotient
//! floor(x * 2^32 / l1) of two integers; `q.div_pos` on the raw integers
//! computes exactly that (it scales its numerator by 2^32). |x| is a sign
//! hint s with m = x (1 - 2s) range-checked to 64 bits, which also proves x an
//! integer (the inputs carry no range proof; `q.max_of([x, -x])` would admit
//! half-integer field elements) and is cheaper than `q.abs`. The window proof
//! of l1 then puts every x in the window. A zero row (l1 = 0, NumPy gives
//! nan) has no proof.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let mut result = Vec::with_capacity(20);
    for row in x.chunks(4) {
        let abs: Vec<Cell> = row.iter().map(|v| abs_int(&q, ctx, *v)).collect();
        let l1 = q.sum(ctx, abs);
        let l1 = q.bound(ctx, l1);
        for v in row {
            result.push(q.div_pos(ctx, *v, l1).cell);
        }
    }
    out.floats(&result);
    Ok(())
}

/// |x| for an input with no range proof: a sign hint s, m = x (1 - 2s), and m
/// in [0, 2^64). Then x = +-m is an integer of magnitude below 2^64 and m = |x|;
/// s is pinned for x != 0 (the other sign gives m = p - |x|, far above 2^64)
/// and irrelevant for x = 0.
fn abs_int(q: &Q32, ctx: &mut Ctx, x: Cell) -> Cell {
    let s = ctx.load_witness(Fr::from((crate::q32::signed(x.value()) < 0) as u64));
    q.gate.assert_bit(ctx, s);
    let sign = q.gate.mul_add(ctx, s, Constant(-Fr::from(2)), Constant(Fr::ONE));
    let m = q.gate.mul(ctx, x, sign);
    q.range.range_check(ctx, m, 64);
    m
}
