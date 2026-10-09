//! DS-1000 case 0365, halo2-base baseline.
//! Kernel: `np.insert(a, pos, element)` on `Float[100]` with `Integer` pos and
//! element (inserted as `float(element)`).
//! NumPy accepts `pos` in [-100, 100] (negative counts from the end) and
//! raises IndexError otherwise (no proof). With `P` the insertion point and
//! `g_k = [P < k]`, output `k` is `a[k]` (k < P), `element` (k = P) or
//! `a[k-1]` (k > P), i.e. `a[k] + g_{k+1} (e - a[k]) - g_k (e - a[k-1])`:
//! two products per output and no comparison. The thermometer `g` is a hint
//! pinned by `insertion_point` (no range check).

use crate::prelude::*;

const N: usize = 100;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, pos, element) = (inputs.floats("a", &[N])?, inputs.int("pos")?, inputs.int("element")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let pos = q.load_int(ctx, pos);
    let element = q.load_int(ctx, element);
    let elem = q.int_to_float(ctx, element);
    q.bound(ctx, elem); // The inserted float must fit the Q32 window.
    let g = insertion_point(&q, ctx, pos, N);
    // d[k] = e - a[k]; out[k] = a[k] + g_{k+1} d[k] - g_k d[k-1], g_0 = 0.
    let d: Vec<Cell> = a.iter().map(|x| q.sub(ctx, elem, *x)).collect();
    let mut result = Vec::with_capacity(N + 1);
    for k in 0..=N {
        let up = if k < N { q.gate.mul_add(ctx, g[k], d[k], a[k]) } else { elem };
        result.push(if k > 0 { q.gate.sub_mul(ctx, up, g[k - 1], d[k - 1]) } else { up });
    }
    out.floats(&result);
    Ok(())
}

/// The insertion point `P` of `np.insert` on an axis of length `n`
/// (`P = pos` for `pos` in [0, n], `pos + n` for `pos` in [-n, -1]) as the
/// thermometer `g[k - 1] = [P < k]`, `k = 1..=n`.
///
/// Hint: the bits `g` and a sign bit `s`, pinned by: `g[k] => g[k + 1]`
/// (monotone, so `g` is the thermometer of its number of zeros `Z`),
/// `Z = pos + n s`, and `s => g[n - 1]` (`s = 1` only with `Z < n`). For
/// `pos` in [0, n] only `s = 0` fits (`s = 1` would need `Z = pos + n >= n`),
/// for `pos` in [-n, -1] only `s = 1` (`s = 0` would need `Z < 0`); any other
/// `pos` gives `Z` outside [0, n]: no witness.
fn insertion_point(q: &Q32, ctx: &mut Ctx, pos: Cell, n: usize) -> Vec<Cell> {
    let v = crate::q32::signed(pos.value());
    let point = if v < 0 { v + n as i128 } else { v };
    let g: Vec<bool> = (1..=n as i128).map(|k| point < k).collect();
    insertion_point_with(q, ctx, pos, &g, v < 0)
}

fn insertion_point_with(q: &Q32, ctx: &mut Ctx, pos: Cell, g: &[bool], neg: bool) -> Vec<Cell> {
    let n = g.len();
    let g = q.load_bools(ctx, g); // hint bits, each constrained to {0, 1}
    let s = q.load_bools(ctx, &[neg])[0];
    for k in 0..n - 1 {
        let both = q.gate.mul(ctx, g[k], g[k + 1]);
        ctx.constrain_equal(&both, &g[k]);
    }
    let s_and_last = q.gate.mul(ctx, s, g[n - 1]);
    ctx.constrain_equal(&s_and_last, &s);
    // sum_k g[k] + n s + pos = n, i.e. Z = n - sum_k g[k] = pos + n s.
    let ones = g.iter().map(|_| Constant(Fr::ONE)).chain([Constant(fe(n as i128))]);
    let weighted = q.gate.inner_product(ctx, g.iter().chain([&s]).map(|c| Existing(*c)), ones);
    let total = q.add(ctx, weighted, pos);
    q.gate.assert_is_const(ctx, &total, &fe(n as i128));
    g
}
