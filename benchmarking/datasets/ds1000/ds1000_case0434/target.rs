//! DS-1000 case 0434, halo2-base baseline.
//! Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on `a: Integer[4, 4]` with
//! `Integer[2]` index arrays.
//! NumPy index rules for each of the two indices: it hits row i when it is i
//! or i - 4, tested once as (idx - i)(idx - i + 4) == 0, and exactly one row
//! must be hit (an index outside [-4, 4) raises, so it has no proof); the same
//! for columns. Row i is kept when neither index hits it,
//! keep_i = (1 - hit0_i)(1 - hit1_i); each row and column test is made once,
//! and cell (i, j) keeps a[i, j] times keep_i times keep_j.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let (zero_rows, zero_cols) = (inputs.ints("zero_rows", &[2])?, inputs.ints("zero_cols", &[2])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let zero_rows = q.load_ints(ctx, &zero_rows);
    let zero_cols = q.load_ints(ctx, &zero_cols);
    let keep_row = kept(&q, ctx, &zero_rows, 4);
    let keep_col = kept(&q, ctx, &zero_cols, 4);
    let mut result = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            let v = q.gate.mul(ctx, a[i * 4 + j], keep_row[i]);
            result.push(q.gate.mul(ctx, v, keep_col[j]));
        }
    }
    out.ints(&result);
    Ok(())
}

/// For each position `p` of an axis of length `len`, 1 if no index in `idxs`
/// hits it. Each index must hit exactly one position (NumPy's wrap and bounds).
fn kept(q: &Q32, ctx: &mut Ctx, idxs: &[Cell], len: usize) -> Vec<Cell> {
    let hits: Vec<Vec<Cell>> = idxs
        .iter()
        .map(|idx| {
            let idx_sq = q.gate.mul(ctx, *idx, *idx);
            let hits: Vec<Cell> = (0..len).map(|p| wraps_to(q, ctx, *idx, idx_sq, p, len)).collect();
            let total = q.sum(ctx, hits.iter().copied());
            q.gate.assert_is_const(ctx, &total, &Fr::ONE);
            hits
        })
        .collect();
    (0..len)
        .map(|p| {
            let not_first = q.gate.not(ctx, hits[0][p]);
            hits[1..].iter().fold(not_first, |keep, h| q.gate.mul_not(ctx, h[p], keep))
        })
        .collect()
}

/// [idx == pos] + [idx == pos - len] (idx is a NumPy index of position `pos` on
/// an axis of length `len`) as one zero test: (idx - pos)(idx - pos + len) = 0,
/// i.e. idx^2 + (len - 2 pos) idx == pos (len - pos). `idx_sq` is idx^2.
fn wraps_to(q: &Q32, ctx: &mut Ctx, idx: Cell, idx_sq: Cell, pos: usize, len: usize) -> Cell {
    let (pos, len) = (pos as i128, len as i128);
    let t = q.gate.mul_add(ctx, idx, Constant(fe(len - 2 * pos)), idx_sq);
    q.eq(ctx, t, Constant(fe(pos * (len - pos))))
}
