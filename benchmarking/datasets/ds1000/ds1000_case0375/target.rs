//! DS-1000 case 0375, halo2-base baseline.
//! Kernel: on `grades: Float[50]` and `threshold: Float`, with `xs = sort(grades)`
//! and `ys = arange(1, 51) / 50`: `t = (ys > threshold).argmax()`, return
//! `[xs[0], xs[t]]`.
//! `ys` is constant, so `t` depends on `threshold` alone. The Q32 constants are
//! `Y_k = round(k / 50 * 2^32) = floor((k * 2^32 + 25) / 50)` (`k = 1..50`), and
//! for the Q32 threshold `T`, `Y_k <= T` iff `k * 2^32 <= 50 T + 24`, i.e.
//! `k <= u = floor((50 T + 24) / 2^32)`. So `clamp(u, 0, 50)` values of `ys` are
//! `<= threshold`, and `argmax` (the first `True`, `0` when there is none) is
//! `t = u` if `0 <= u < 50`, else `0`.
//! `xs[0]` is `q.min_of`. `xs[t]` is an order statistic at a private position:
//! a hint pinned by counting (see `kth_smallest`): 50 comparisons and 50
//! equality tests, where a sort would need a permutation argument on top of 49
//! comparisons.

use crate::prelude::*;
use crate::q32::signed;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (grades, threshold) = (inputs.floats("grades", &[50])?, inputs.float("threshold")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let grades_bounded = q.load_floats_bounded(ctx, &grades);
    let grades: Vec<Cell> = grades_bounded.iter().map(|g| g.cell).collect();
    let threshold = q.load_float(ctx, threshold);

    let low = q.min_of(ctx, &grades);

    let raw = q.gate.mul_add(ctx, threshold, Constant(fe(50)), Constant(fe(24)));
    let u = q.rescale(ctx, raw);
    let negative = q.is_neg(ctx, u);
    let below_len = q.lt_const(ctx, u, 50);
    let in_range = q.gate.mul_not(ctx, negative, below_len);
    let t = q.gate.mul(ctx, in_range, u.cell);

    let high = kth_smallest(&q, ctx, &grades_bounded, t);
    out.floats(&[low, high.cell]);
    Ok(())
}

/// The value at position `k` (0-based) of `xs` sorted ascending, for a `k` the
/// circuit already knows to lie in `[0, xs.len())`.
///
/// Hint `v` (in the window), pinned by `#{x < v} <= k < #{x <= v}`: if `v < v'`
/// both satisfied it, `#{x <= v} <= #{x < v'} <= k < #{x <= v}`, so at most one
/// `v` does, and the sorted value at position `k` does. That `v` is one of the
/// `xs` (`#{x == v} > 0`). The two inequalities are 8-bit range checks of
/// `k - #{x < v}` and `#{x <= v} - k - 1`, whose integer values lie in
/// `(-128, 128)`; a negative one is a field element above `p - 128` and fails.
fn kth_smallest(q: &Q32, ctx: &mut Ctx, xs: &[I64], k: Cell) -> I64 {
    assert!(xs.len() < 128);
    let mut sorted: Vec<i128> = xs.iter().map(|x| x.value()).collect();
    sorted.sort();
    let pos = signed(k.value()).clamp(0, xs.len() as i128 - 1) as usize;
    let v = ctx.load_witness(fe(sorted[pos]));
    let v = q.bound(ctx, v);

    let below: Vec<Cell> = xs.iter().map(|x| q.lt(ctx, *x, v)).collect();
    let equal: Vec<Cell> = xs.iter().map(|x| q.eq(ctx, x.cell, v.cell)).collect();
    let n_below = q.sum(ctx, below);
    let n_equal = q.sum(ctx, equal);
    let slack_low = q.sub(ctx, k, n_below); // k - #{x < v}
    q.range.range_check(ctx, slack_low, 8);
    let slack_high = q.sub(ctx, n_equal, slack_low); // #{x <= v} - k
    let slack_high = q.sub(ctx, slack_high, Constant(Fr::ONE));
    q.range.range_check(ctx, slack_high, 8);
    v
}
