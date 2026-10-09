//! DS-1000 case 0346, halo2-base baseline, literal translation of kernel.py
//! (LITERAL_BASELINES.md).
//! Kernel:
//!   a[row-1, :] *= multiply_number
//!   result = np.cumsum(a[row-1, :])
//! on `a: Float[8, 6]`, `row: Integer`, `multiply_number: Float`. The row
//! analog of case 0345: NumPy index semantics require `row - 1` in `[-8, 8)`
//! (negative indices count from the end); any other value has no witness.
//! `row - 1` is written twice in the source, so it is computed, and its
//! validity checked, twice (no CSE across statements). There is no shared
//! gadget for "write at a private index" (only reads), so the write uses the
//! local helper `row_hits` below, per the L5 rule for `a[i] = v`. The write
//! touches every one of the 48 cells of `a` (L2: the full mutated array),
//! selecting per row and column whether that row is the target row. The read
//! (`a[row-1, :]`) is the direct form of an indexed read (`q.index`), done
//! once per column (each column is an 8-element array indexed by `row - 1`).
//! `np.cumsum` is a running accumulator over the 6 columns in order (L5).

use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[ROWS, COLS])?;
    let row = inputs.int("row")?;
    let multiply_number = inputs.float("multiply_number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let row = q.load_int(ctx, row);
    let multiply_number = q.load_float(ctx, multiply_number);

    // a[row-1, :] *= multiply_number
    let idx1 = q.gate.sub(ctx, row, Constant(Fr::ONE));
    let hits1 = row_hits(&q, ctx, idx1, ROWS);
    for i in 0..ROWS {
        for j in 0..COLS {
            let old = a[i * COLS + j];
            let scaled = q.mul(ctx, old, multiply_number).cell;
            a[i * COLS + j] = q.gate.select(ctx, scaled, old, hits1[i]);
        }
    }

    // result = np.cumsum(a[row-1, :])
    let idx2 = q.gate.sub(ctx, row, Constant(Fr::ONE)); // recomputed: no CSE across statements
    let mut acc: Option<Cell> = None;
    let mut cumsum = Vec::with_capacity(COLS);
    for j in 0..COLS {
        let column: Vec<Cell> = (0..ROWS).map(|i| a[i * COLS + j]).collect();
        let picked = q.index(ctx, &column, idx2);
        acc = Some(match acc {
            None => picked,
            Some(prev) => q.add(ctx, prev, picked),
        });
        cumsum.push(acc.unwrap());
    }
    out.floats(&cumsum);
    Ok(())
}

/// The hit vector of a NumPy row index `idx` over `len` rows, for a
/// private-index write (no shared gadget writes by a private index): row `i`
/// is hit when `idx` is `i` or `i - len` (the wrapped negative index), i.e.
/// when `(idx - i) * (idx - i + len)` is zero (a field has no zero divisors),
/// one zero test per row; asserting exactly one hit pins `idx` to a valid
/// NumPy index (any other value has no witness, as NumPy raises
/// `IndexError`). Shared across the `COLS` columns the write touches, since
/// the source statement applies the one row index to every column.
fn row_hits(q: &Q32, ctx: &mut Ctx, idx: Cell, len: usize) -> Vec<Cell> {
    let n = len as i128;
    let hits: Vec<Cell> = (0..n)
        .map(|i| {
            let at = q.gate.sub(ctx, idx, Constant(fe(i)));
            let wrapped = q.gate.add(ctx, at, Constant(fe(n)));
            let both = q.gate.mul(ctx, at, wrapped);
            q.gate.is_zero(ctx, both)
        })
        .collect();
    let total = q.gate.sum(ctx, hits.iter().map(|h| Existing(*h)));
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hits
}
