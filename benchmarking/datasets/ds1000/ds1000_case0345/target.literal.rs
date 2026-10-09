//! DS-1000 case 0345, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel:
//!   a[:, col-1] *= multiply_number
//!   result = np.cumsum(a[:, col-1])
//! on `a: Float[8, 6]`, `col: Integer`, `multiply_number: Float`. NumPy index
//! semantics: `col - 1` must lie in `[-6, 6)` (negative indices count from the
//! end); any other value has no witness. `col - 1` is written twice in the
//! source (once per statement), so it is computed, and its validity checked,
//! twice (no CSE across statements). There is no shared gadget for "write at
//! a private index" (only reads), so the write uses the local helper
//! `column_hits` below, per the L5 rule for `a[i] = v`. The write touches
//! every one of the 48 cells of `a` (L2: the full mutated array), selecting
//! per row and column whether that column is the target column. The read
//! (`a[:, col-1]`) is the direct form of an indexed read (`q.index`).
//! `np.cumsum` is a running accumulator over the 8 rows in order (L5).

use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[ROWS, COLS])?;
    let col = inputs.int("col")?;
    let multiply_number = inputs.float("multiply_number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let col = q.load_int(ctx, col);
    let multiply_number = q.load_float(ctx, multiply_number);

    // a[:, col-1] *= multiply_number
    let idx1 = q.gate.sub(ctx, col, Constant(Fr::ONE));
    let hits1 = column_hits(&q, ctx, idx1, COLS);
    for i in 0..ROWS {
        for k in 0..COLS {
            let old = a[i * COLS + k];
            let scaled = q.mul(ctx, old, multiply_number).cell;
            a[i * COLS + k] = q.gate.select(ctx, scaled, old, hits1[k]);
        }
    }

    // result = np.cumsum(a[:, col-1])
    let idx2 = q.gate.sub(ctx, col, Constant(Fr::ONE)); // recomputed: no CSE across statements
    let mut acc: Option<Cell> = None;
    let mut cumsum = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let row = &a[i * COLS..(i + 1) * COLS];
        let picked = q.index(ctx, row, idx2);
        acc = Some(match acc {
            None => picked,
            Some(prev) => q.add(ctx, prev, picked),
        });
        cumsum.push(acc.unwrap());
    }
    out.floats(&cumsum);
    Ok(())
}

/// The hit vector of a NumPy column index `idx` over `len` columns, for a
/// private-index write (no shared gadget writes by a private index): column
/// `k` is hit when `idx` is `k` or `k - len` (the wrapped negative index),
/// i.e. when `(idx - k) * (idx - k + len)` is zero (a field has no zero
/// divisors), one zero test per column; asserting exactly one hit pins `idx`
/// to a valid NumPy index (any other value has no witness, as NumPy raises
/// `IndexError`). Shared across the `ROWS` rows the write touches, since the
/// source statement applies the one column index to every row.
fn column_hits(q: &Q32, ctx: &mut Ctx, idx: Cell, len: usize) -> Vec<Cell> {
    let n = len as i128;
    let hits: Vec<Cell> = (0..n)
        .map(|k| {
            let at = q.gate.sub(ctx, idx, Constant(fe(k)));
            let wrapped = q.gate.add(ctx, at, Constant(fe(n)));
            let both = q.gate.mul(ctx, at, wrapped);
            q.gate.is_zero(ctx, both)
        })
        .collect();
    let total = q.gate.sum(ctx, hits.iter().map(|h| Existing(*h)));
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hits
}
