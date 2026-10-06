//! DS-1000 case 0300, halo2-base baseline.
//! Kernel: `np.percentile(a, p)` on `Float[20]` with an `Integer` p.
//! NumPy rejects p outside [0, 100] (ValueError: no proof). Default method
//! 'linear': with `19 * p = 100 * lo + r` (exact integers, `lo` in [0, 19]),
//! the result is `s[lo] + (r / 100) * (s[hi] - s[lo])` for `s = sorted(a)`,
//! `hi = min(lo + 1, 19)`; the product is `floor(r * (s[hi] - s[lo]) / 100)`,
//! within one Q32 unit of the exact value (NumPy's `_lerp` differs from it
//! by double rounding only).
//! Only two order statistics are needed, so no sort: element `i` is the key
//! `32 * a_i + 2^68 + i` (`a_i` bounded to the window: distinct integers
//! ordered by `(a_i, i)`), and the key at rank `m` is a hint pinned as the
//! only key with exactly `m` keys below it (`order_statistic`). Two ranks
//! cost 40 comparisons; a sorting network for 20 values needs 97
//! compare-exchanges.

use crate::prelude::*;
use crate::q32::signed;

const N: usize = 20;
/// Shifts `32 * a_i` (`a_i` in the window) into `[0, 2^69)`.
const KEY_SHIFT: i128 = 1 << 68;
const KEY_BITS: usize = 69;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, p) = (inputs.floats("a", &[N])?, inputs.int("p")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a); // each a_i in the window, so each key is an integer
    let p = q.load_int(ctx, p);
    let keys: Vec<Cell> = a
        .iter()
        .enumerate()
        .map(|(i, x)| {
            q.gate.mul_add(ctx, *x, Constant(fe(32)), Constant(fe(KEY_SHIFT + i as i128)))
        })
        .collect();

    q.range.check_less_than_safe(ctx, p, 101);
    let g = q.gate.mul(ctx, p, Constant(fe(19)));
    let (lo, r) = q.divmod_small(ctx, g, 100, 11); // g <= 1900 < 2^11
    let last = q.eq(ctx, lo, Constant(fe(N as i128 - 1)));
    let hi = q.add(ctx, lo, Constant(Fr::ONE));
    let hi = q.sub(ctx, hi, last);

    let s_lo = order_statistic(&q, ctx, &keys, lo);
    let s_hi = order_statistic(&q, ctx, &keys, hi);
    let diff = q.sub(ctx, s_hi, s_lo);
    let scaled = q.gate.mul(ctx, r, diff);
    let (frac, _) = q.floor_div_const(ctx, scaled, 100);
    let result = q.add(ctx, s_lo, frac.cell);
    out.float(result);
    Ok(())
}

/// The value of rank `rank` (0-based, ascending) among the keys
/// `32 * a_i + 2^68 + i`. Hint: the key `t` of that rank, pinned by
/// `prod_i (t - key_i) = 0` (`t` is a key) and "exactly `rank` keys are below
/// `t`" (the keys are distinct, so one key has each rank). The value is the
/// quotient of `t - 2^68` by 32 (`floor_div_const`: quotient in the window,
/// remainder the index).
fn order_statistic(q: &Q32, ctx: &mut Ctx, keys: &[Cell], rank: Cell) -> Cell {
    let mut values: Vec<i128> = keys.iter().map(|k| signed(k.value())).collect();
    values.sort_unstable();
    let t = values.get(signed(rank.value()) as usize).copied().unwrap_or(0);
    order_statistic_with(q, ctx, keys, rank, t)
}

fn order_statistic_with(q: &Q32, ctx: &mut Ctx, keys: &[Cell], rank: Cell, t: i128) -> Cell {
    let t = ctx.load_witness(fe(t));
    let mut prod = q.sub(ctx, t, keys[0]);
    for key in &keys[1..] {
        let diff = q.sub(ctx, t, *key);
        prod = q.gate.mul(ctx, prod, diff);
    }
    q.gate.assert_is_const(ctx, &prod, &Fr::ZERO);
    let below: Vec<Cell> = keys.iter().map(|k| q.range.is_less_than(ctx, *k, t, KEY_BITS)).collect();
    let below = q.sum(ctx, below);
    ctx.constrain_equal(&below, &rank);
    let shifted = q.sub(ctx, t, Constant(fe(KEY_SHIFT)));
    q.floor_div_const(ctx, shifted, 32).0.cell
}
