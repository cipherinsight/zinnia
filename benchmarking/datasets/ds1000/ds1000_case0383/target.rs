//! DS-1000 case 0383, halo2-base baseline.
//! Kernel: `N = 7; np.argsort(a)[::-1][:N]` on `Float[100]`: the indices of the
//! 7 largest values, largest first.
//! Element `i` is the key `128 * a_i + 2^70 + i` (`a_i` bounded to the window:
//! distinct integers in `[0, 2^71)` ordered by `(a_i, i)`), so the result is
//! the reversed stable argsort: among equal values the larger index comes
//! first (NumPy's default introsort leaves the order of equal values
//! unspecified). The 7 largest keys are a hint pinned by checks (`top_keys`):
//! 100 comparisons against the 7th, 7 membership products and 6 order checks,
//! about half the cells of a top-7 selection network (756 compare-exchanges
//! pruned from a 100-key merge exchange). The index of a key is its remainder
//! mod 128 (`floor_div_const`, quotient pinned to the window).

use crate::prelude::*;
use crate::q32::signed;

const N: usize = 100;
const TOP: usize = 7;
/// Shifts `128 * a_i` (`a_i` in the window) into `[0, 2^71)`.
const KEY_SHIFT: i128 = 1 << 70;
const KEY_BITS: usize = 71;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let keys: Vec<Cell> = a
        .iter()
        .enumerate()
        .map(|(i, x)| {
            q.bound(ctx, *x); // a_i in the window, so the key is an integer
            q.gate.mul_add(ctx, *x, Constant(fe(128)), Constant(fe(KEY_SHIFT + i as i128)))
        })
        .collect();
    let top = top_keys(&q, ctx, &keys, TOP);
    let indices: Vec<Cell> = top
        .iter()
        .map(|key| {
            let shifted = q.sub(ctx, *key, Constant(fe(KEY_SHIFT)));
            q.floor_div_const(ctx, shifted, 128).1
        })
        .collect();
    out.ints(&indices);
    Ok(())
}

/// The `k` largest of `keys` (distinct integers in `[0, 2^71)`), largest
/// first. Hint: their values `t_1 > ... > t_k`, pinned by
/// 1. `t_{s+1} < t_s` for each `s` (`check_less_than`);
/// 2. each `t_s` is one of the keys: `prod_i (t_s - key_i) = 0`;
/// 3. exactly `n - k` keys are below `t_k`.
///
/// By 1 and 2 the `t_s` are `k` distinct keys `>= t_k`; by 3 no other key is
/// `>= t_k`. So they are the `k` largest keys, in decreasing order.
fn top_keys(q: &Q32, ctx: &mut Ctx, keys: &[Cell], k: usize) -> Vec<Cell> {
    let mut values: Vec<i128> = keys.iter().map(|c| signed(c.value())).collect();
    values.sort_unstable_by(|x, y| y.cmp(x));
    top_keys_with(q, ctx, keys, &values[..k])
}

fn top_keys_with(q: &Q32, ctx: &mut Ctx, keys: &[Cell], hint: &[i128]) -> Vec<Cell> {
    let top: Vec<Cell> = hint.iter().map(|v| ctx.load_witness(fe(*v))).collect();
    for pair in top.windows(2) {
        q.range.check_less_than(ctx, pair[1], pair[0], KEY_BITS);
    }
    for t in &top {
        let mut prod = q.sub(ctx, *t, keys[0]);
        for key in &keys[1..] {
            let diff = q.sub(ctx, *t, *key);
            prod = q.gate.mul(ctx, prod, diff);
        }
        q.gate.assert_is_const(ctx, &prod, &Fr::ZERO);
    }
    let last = *top.last().expect("k > 0");
    let below: Vec<Cell> = keys.iter().map(|key| q.range.is_less_than(ctx, *key, last, KEY_BITS)).collect();
    let below = q.sum(ctx, below);
    q.gate.assert_is_const(ctx, &below, &Fr::from((keys.len() - hint.len()) as u64));
    top
}
