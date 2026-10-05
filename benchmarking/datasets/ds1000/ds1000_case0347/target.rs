//! DS-1000 case 0347, halo2-base baseline.
//! Kernel: `a[row-1, :] /= divide_number; np.multiply.reduce(a[row-1, :])` on
//! `a: Float[8, 6]`, `row: Integer`, `divide_number: Float`.
//! Only the selected row reaches the output: it is gathered once (one-hot over
//! the 8 rows, built once and reused for the 6 columns; NumPy wrap and bounds
//! for `row - 1`), divided by `divide_number` (6 float divisions sharing one
//! sign test of the divisor; a zero divisor has no proof) and multiplied out
//! left to right (5 rescaled products).

use crate::prelude::*;
use crate::q32::signed;

const ROWS: usize = 8;
const COLS: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[ROWS, COLS])?;
    let row = inputs.int("row")?;
    let divide_number = inputs.float("divide_number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let row = q.load_int(ctx, row);
    let divide_number = q.load_float(ctx, divide_number);

    let hit = position_onehot(&q, ctx, row, 1, ROWS);
    let divisor = q.bound(ctx, divide_number);
    let divisor_abs = q.abs(ctx, divisor);
    let rem_limit = q.gate.sub(ctx, divisor_abs, Constant(Fr::ONE)); // |b| - 1
    let quotients: Vec<Cell> = (0..COLS)
        .map(|j| {
            let column = (0..ROWS).map(|i| Existing(a[i * COLS + j]));
            let picked = q.gate.inner_product(ctx, hit.iter().map(|h| Existing(*h)), column);
            div_by(&q, ctx, picked, divisor.cell, rem_limit).cell
        })
        .collect();
    let product = quotients[1..].iter().fold(quotients[0], |acc, x| q.mul(ctx, acc, *x).cell);
    out.float(product);
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

/// `q.div` for a divisor `b` (in the window) whose `rem_limit = |b| - 1` the
/// caller computed once: `floor(a * 2^32 / b)` with remainder in `[0, |b|)`.
/// Constraints, as in `q.div`: `a * 2^32 = quot * b + rem`, `rem` in
/// `[0, 2^64)`, `|b| - 1 - rem` in `[0, 2^64)` (so `rem < |b|` and `b != 0`),
/// `quot` in the window. With `|b| <= 2^63` every term is below `2^127`, so
/// the identity holds over the integers and fixes `(quot, rem)`.
fn div_by(q: &Q32, ctx: &mut Ctx, a: Cell, b: Cell, rem_limit: Cell) -> I64 {
    let num = signed(a.value()) * SCALE;
    let bv = signed(b.value());
    let (quot, rem) = if bv == 0 { (0, 0) } else { (num.div_euclid(bv), num.rem_euclid(bv)) };
    let scaled = q.gate.mul(ctx, a, Constant(fe(SCALE)));
    let quot = ctx.load_witness(fe(quot));
    let rem = ctx.load_witness(fe(rem));
    let recomposed = q.gate.mul_add(ctx, quot, b, rem);
    ctx.constrain_equal(&recomposed, &scaled);
    q.range.range_check(ctx, rem, 64);
    let gap = q.gate.sub(ctx, rem_limit, rem);
    q.range.range_check(ctx, gap, 64);
    q.bound(ctx, quot)
}
