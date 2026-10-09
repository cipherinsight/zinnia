//! DS-1000 case 0450, halo2-base literal-translation baseline
//! (LITERAL_BASELINES.md).
//! Kernel: `result = a[:, np.array(second).reshape(-1, 1), third]` on
//! `Float[7, 8, 9]` with `second`, `third` `Integer[2]`: NumPy broadcasts
//! `second` (reshaped to `(2, 1)`) against `third` (`(2,)`) over axes 1 and
//! 2, so `result[i, j, k] = a[i, second[j], third[k]]`, shape `(7, 2, 2)`.
//! Negative indices wrap (NumPy rule); any other out-of-range index raises
//! `IndexError` (no satisfying witness).
//!
//! Translation choices:
//! - The one statement reads two private indices at once, one per axis.
//!   Translated axis by axis in the order written, `a[..., second[j],
//!   ...]` then `[..., third[k]]`: for each `i`, `j`, gather the row
//!   `a[i, second[j], :]` (9 elements, one indexed read per position over
//!   the 8 candidates of axis 1), then read `third[k]` out of that row
//!   (one indexed read over its 9 positions). Nothing is cached across `i`
//!   or `j`: each read is a fresh call, as the source has no caching
//!   across the broadcast (the optimized baseline resolves each index
//!   once and reuses it for every `i`, which its own header names as the
//!   optimization this baseline does not take).
//! - Axis 1 (length 8) uses `q.index` directly: it is at most
//!   `ONEHOT_MAX_LEN`, so `q.index` takes its one-hot branch, which needs
//!   no extra window check of `second[j]`.
//! - Axis 2 (length 9) cannot use `q.index`: beyond `ONEHOT_MAX_LEN` it
//!   takes the branch that calls `q.bound`, which would check `third[k]`'s
//!   window a second time (it is already checked once when loaded),
//!   failing the "each input checked once" requirement (L7). L6 allows a
//!   local helper when no non-forbidden library gadget avoids this;
//!   `index_long` below is the one-hot-plus-linear-identity decode that
//!   needs no range check at all (sound because the index is already a
//!   window-checked int64 from the load: see its doc comment), written
//!   locally rather than by calling the forbidden `q.index_onehot`/
//!   `q.index_i64` by name.

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

    // result[i, j, k] = a[i, second[j], third[k]]
    let mut result = Vec::with_capacity(D0 * 2 * 2);
    for i in 0..D0 {
        for j in 0..2 {
            // row = a[i, second[j], :]
            let row: Vec<Cell> = (0..D2)
                .map(|p2| {
                    let candidates: Vec<Cell> = (0..D1).map(|p1| a[(i * D1 + p1) * D2 + p2]).collect();
                    q.index(ctx, &candidates, second[j])
                })
                .collect();
            for k in 0..2 {
                result.push(index_long(&q, ctx, &row, third[k]));
            }
        }
    }
    out.floats(&result);
    Ok(())
}

/// `arr[idx]` for a private index `idx` on an axis longer than
/// `ONEHOT_MAX_LEN`, without a second window check of `idx` (`idx` is
/// already checked once when loaded as an input): the one-hot position
/// `e[p] = [idx selects p]` (`p` in `[0, n)` for `idx` in `[0, n)`,
/// `p = idx + n` for `idx` in `[-n, -1]`, `n = arr.len()`) is pinned by a
/// hint and a linear identity, with no range check needed. The identity
/// `sum_p p e[p] - n s = idx` with `e` one-hot and `s` a bit has exactly
/// one solution when `idx` is a genuine window-checked int64 (no field
/// wraparound between the tiny values `p`, `p - n` and the huge window),
/// so any `idx` outside `[-n, n)` has no witness (`IndexError`).
fn index_long(q: &Q32, ctx: &mut Ctx, arr: &[Cell], idx: Cell) -> Cell {
    let n = arr.len();
    let v = crate::q32::signed(idx.value());
    let point = if v < 0 { v + n as i128 } else { v };
    let e: Vec<bool> = (0..n as i128).map(|p| p == point).collect();
    let neg = v < 0;
    let e = q.load_bools(ctx, &e);
    let s = q.load_bools(ctx, &[neg])[0];
    let total = q.sum(ctx, e.iter().copied());
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    let weights = (1..n).map(|p| Constant(fe(p as i128))).chain([Constant(-fe(n as i128))]);
    let signed_pos = q.gate.inner_product(ctx, e[1..].iter().chain([&s]).map(|c| Existing(*c)), weights);
    ctx.constrain_equal(&signed_pos, &idx);
    q.gate.inner_product(ctx, e.iter().map(|c| Existing(*c)), arr.iter().map(|c| Existing(*c)))
}
