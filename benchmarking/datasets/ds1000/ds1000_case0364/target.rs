//! DS-1000 case 0364, halo2-base baseline.
//! Kernel: `np.insert(a, pos, element, axis=0)` on `Float[100, 10]` with an
//! `Integer` pos and a `Float[10]` row.
//! NumPy accepts `pos` in [-100, 100] (negative counts from the end) and
//! raises IndexError otherwise (no proof). With `P` the insertion point and
//! `g_k = [P < k]`, output row `k` is `a[k]` (k < P), `element` (k = P) or
//! `a[k-1]` (k > P), i.e. `a[k] + g_{k+1} (e - a[k]) - g_k (e - a[k-1])`
//! per cell: two products per output cell and no comparison. The thermometer
//! `g` is a hint pinned by `insertion_point` (no range check).

use crate::prelude::*;

const N: usize = 100;
const W: usize = 10;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, pos, element) = (inputs.floats("a", &[N, W])?, inputs.int("pos")?, inputs.floats("element", &[W])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let pos = q.load_int(ctx, pos);
    let elem = q.load_floats(ctx, &element);
    let g = insertion_point(&q, ctx, pos, N);
    // Per column c: d[k] = e[c] - a[k][c]; out[k][c] = a[k][c] + g_{k+1} d[k] - g_k d[k-1], g_0 = 0.
    let d: Vec<Cell> = (0..N * W).map(|kc| q.sub(ctx, elem[kc % W], a[kc])).collect();
    let mut result = Vec::with_capacity((N + 1) * W);
    for k in 0..=N {
        for c in 0..W {
            let up = if k < N { q.gate.mul_add(ctx, g[k], d[k * W + c], a[k * W + c]) } else { elem[c] };
            result.push(if k > 0 { q.gate.sub_mul(ctx, up, g[k - 1], d[(k - 1) * W + c]) } else { up });
        }
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
