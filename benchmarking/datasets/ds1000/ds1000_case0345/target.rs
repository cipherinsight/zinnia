//! DS-1000 case 0345, halo2-base baseline.
//! Kernel: `a[:, col-1] *= multiply_number; np.cumsum(a[:, col-1])` on
//! `a: Float[8, 6]`, `col: Integer`, `multiply_number: Float`.
//! Only the selected column reaches the output: it is gathered once, scaled
//! (8 rescaled products) and summed (exact prefix sums). The index `col - 1`
//! follows NumPy (`-6 <= col - 1 < 6`, negative values wrap, any other `col`
//! has no proof). Its one-hot over the 6 columns is built once and reused for
//! the 8 rows.

use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[ROWS, COLS])?;
    let col = inputs.int("col")?;
    let multiply_number = inputs.float("multiply_number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let col = q.load_int(ctx, col);
    let multiply_number = q.load_float(ctx, multiply_number);

    let hit = position_onehot(&q, ctx, col, 1, COLS);
    let scaled: Vec<Cell> = a
        .chunks(COLS)
        .map(|row| {
            let picked = q.gate.inner_product(ctx, hit.iter().map(|h| Existing(*h)), row.iter().map(|x| Existing(*x)));
            q.mul(ctx, picked, multiply_number).cell
        })
        .collect();
    let cumsum: Vec<Cell> = q.gate.partial_sums(ctx, scaled.iter().map(|x| Existing(*x))).collect();
    out.floats(&cumsum);
    Ok(())
}

/// The one-hot of the position a NumPy index `idx - shift` selects on an axis
/// of `len` elements, by the construction of `q.index_onehot`, returned so
/// that every gather by the same index shares it: position `p` is hit when
/// `idx` is `u = p + shift` or `v = p + shift - len` (the wrapped negative
/// index), i.e. when `(idx - u) * (idx - v)` is zero (no zero divisors in the
/// field), and the assertion that exactly one position is hit pins `idx` to
/// one of the `2 * len` valid values: any other index has no witness, as
/// NumPy raises `IndexError`. The product is `idx * (idx - u - v) + u * v`,
/// one `mul_add` with a constant.
fn position_onehot(q: &Q32, ctx: &mut Ctx, idx: Cell, shift: i128, len: usize) -> Vec<Cell> {
    let n = len as i128;
    let hit: Vec<Cell> = (0..n)
        .map(|p| {
            let (u, v) = (p + shift, p + shift - n);
            let shifted = q.gate.sub(ctx, idx, Constant(fe(u + v)));
            let prod = q.gate.mul_add(ctx, idx, shifted, Constant(fe(u * v)));
            q.gate.is_zero(ctx, prod)
        })
        .collect();
    let total = q.gate.sum(ctx, hit.iter().map(|h| Existing(*h)));
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hit
}
