//! DS-1000 case 0433, halo2-base baseline.
//! Kernel: `a[zero_rows, :] = 0; a[:, zero_cols] = 0` on `a: Integer[4, 4]` with
//! scalar indices.
//! NumPy index rules: row i is hit when zero_rows is i or i - 4, tested once as
//! (zero_rows - i)(zero_rows - i + 4) == 0, and exactly one row must be hit (an
//! index outside [-4, 4) raises, so it has no proof); the same for columns.
//! Each row and column test is made once; cell (i, j) keeps a[i, j] times
//! (1 - row_i)(1 - col_j).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let (zero_rows, zero_cols) = (inputs.int("zero_rows")?, inputs.int("zero_cols")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_ints(ctx, &a);
    let zero_rows = q.load_int(ctx, zero_rows);
    let zero_cols = q.load_int(ctx, zero_cols);
    let row_hit = one_hot(&q, ctx, zero_rows, 4);
    let col_hit = one_hot(&q, ctx, zero_cols, 4);
    let mut result = Vec::with_capacity(16);
    for i in 0..4 {
        for j in 0..4 {
            let kept = q.gate.mul_not(ctx, row_hit[i], a[i * 4 + j]);
            result.push(q.gate.mul_not(ctx, col_hit[j], kept));
        }
    }
    out.ints(&result);
    Ok(())
}

/// Whether `idx` is each position of an axis of length `len`, with the
/// assertion that exactly one holds (NumPy's wrap and bounds).
fn one_hot(q: &Q32, ctx: &mut Ctx, idx: Cell, len: usize) -> Vec<Cell> {
    let idx_sq = q.gate.mul(ctx, idx, idx);
    let hits: Vec<Cell> = (0..len).map(|p| wraps_to(q, ctx, idx, idx_sq, p, len)).collect();
    let total = q.sum(ctx, hits.iter().copied());
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hits
}

/// [idx == pos] + [idx == pos - len] (idx is a NumPy index of position `pos` on
/// an axis of length `len`) as one zero test: (idx - pos)(idx - pos + len) = 0,
/// i.e. idx^2 + (len - 2 pos) idx == pos (len - pos). `idx_sq` is idx^2.
fn wraps_to(q: &Q32, ctx: &mut Ctx, idx: Cell, idx_sq: Cell, pos: usize, len: usize) -> Cell {
    let (pos, len) = (pos as i128, len as i128);
    let t = q.gate.mul_add(ctx, idx, Constant(fe(len - 2 * pos)), idx_sq);
    q.eq(ctx, t, Constant(fe(pos * (len - pos))))
}
