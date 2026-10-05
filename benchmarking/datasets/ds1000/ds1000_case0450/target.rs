//! DS-1000 case 0450, halo2-base baseline.
//! Kernel: `a[:, np.array(second).reshape(-1, 1), third]` on `Float[7, 8, 9]`
//! with `second`, `third` `Integer[2]`: `result[i, j, k] = a[i, second[j],
//! third[k]]`, shape (7, 2, 2), NumPy index rules (negative indices wrap,
//! others out of range raise IndexError: no proof).
//! Each of the four indices gets its one-hot once (`onehot_index`, a hint
//! pinned without range checks), reused for every `i`: first the `third`
//! axis is contracted (`t[k][p] = a[i, p, third[k]]`), then the `second`
//! axis.

use crate::prelude::*;

const D0: usize = 7;
const D1: usize = 8;
const D2: usize = 9;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[D0, D1, D2])?;
    let (second, third) = (inputs.ints("second", &[2])?, inputs.ints("third", &[2])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let second = q.load_ints(ctx, &second);
    let third = q.load_ints(ctx, &third);
    let at_second: Vec<Vec<Cell>> = second.iter().map(|x| onehot_index(&q, ctx, *x, D1)).collect();
    let at_third: Vec<Vec<Cell>> = third.iter().map(|x| onehot_index(&q, ctx, *x, D2)).collect();

    let mut result = Vec::with_capacity(D0 * 4);
    for i in 0..D0 {
        // t[k][p] = a[i, p, third[k]]
        let t: Vec<Vec<Cell>> = at_third
            .iter()
            .map(|e| {
                (0..D1)
                    .map(|p| {
                        let row = &a[(i * D1 + p) * D2..(i * D1 + p + 1) * D2];
                        q.gate.select_by_indicator(ctx, row.iter().map(|c| Existing(*c)), e.iter().copied())
                    })
                    .collect()
            })
            .collect();
        for e in &at_second {
            for tk in &t {
                result.push(q.gate.select_by_indicator(ctx, tk.iter().map(|c| Existing(*c)), e.iter().copied()));
            }
        }
    }
    out.floats(&result);
    Ok(())
}

/// One-hot `e[p] = [P == p]` of the position `P` a NumPy index `idx` selects
/// on an axis of length `n` (`P = idx` for `idx` in [0, n), `idx + n` for
/// `idx` in [-n, -1]).
///
/// Hint: the bits `e` and a sign bit `s`, pinned by `sum_p e[p] = 1` (one-hot)
/// and `sum_p p e[p] - n s = idx`, i.e. `idx = P - n s` with `P` in [0, n).
/// `s = 0` gives `idx` in [0, n), `s = 1` gives `idx` in [-n, -1]: the two
/// ranges are disjoint, so every valid `idx` has exactly one witness and any
/// other index none (IndexError). No range check.
fn onehot_index(q: &Q32, ctx: &mut Ctx, idx: Cell, n: usize) -> Vec<Cell> {
    let v = crate::q32::signed(idx.value());
    let point = if v < 0 { v + n as i128 } else { v };
    let e: Vec<bool> = (0..n as i128).map(|p| p == point).collect();
    onehot_index_with(q, ctx, idx, &e, v < 0)
}

fn onehot_index_with(q: &Q32, ctx: &mut Ctx, idx: Cell, e: &[bool], neg: bool) -> Vec<Cell> {
    let n = e.len();
    let e = q.load_bools(ctx, e); // hint bits, each constrained to {0, 1}
    let s = q.load_bools(ctx, &[neg])[0];
    let total = q.sum(ctx, e.iter().copied());
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    let weights = (1..n).map(|p| Constant(fe(p as i128))).chain([Constant(-fe(n as i128))]);
    let signed_pos = q.gate.inner_product(ctx, e[1..].iter().chain([&s]).map(|c| Existing(*c)), weights);
    ctx.constrain_equal(&signed_pos, &idx);
    e
}
