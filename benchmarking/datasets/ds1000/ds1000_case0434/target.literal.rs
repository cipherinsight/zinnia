//! DS-1000 case 0434: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each write tests both signed spellings of every axis position and enforces one hit; complete row writes before column writes.
use crate::prelude::*;

// Valid NumPy indices for position k are k and k-4. No ordered
// comparison is part of this write: exact equality handles wrap and bounds.
fn write_hits(q: &Q32, ctx: &mut Ctx, idx: Cell) -> Vec<Cell> {
    let hits: Vec<Cell> = (0..4).map(|k| {
        let positive = q.eq(ctx, idx, Constant(fe(k)));
        let negative = q.eq(ctx, idx, Constant(fe(k - 4)));
        q.gate.or(ctx, positive, negative)
    }).collect();
    let total = q.sum(ctx, hits.iter().copied());
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hits
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("a", &[4, 4])?;
    let rows = inputs.ints("zero_rows", &[2])?;
    let cols = inputs.ints("zero_cols", &[2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut a = q.load_ints(ctx, &av);
    let rows = q.load_ints(ctx, &rows);
    let cols = q.load_ints(ctx, &cols);
    let zero = q.int_const(ctx, 0);
    // a[zero_rows, :] = 0
    for i in 0..2 {
        let hits = write_hits(&q, ctx, rows[i]);
        for row in 0..4 {
            let hit = hits[row];
            for col in 0..4 { a[row * 4 + col] = q.gate.select(ctx, zero, a[row * 4 + col], hit); }
        }
    }
    // a[:, zero_cols] = 0
    for i in 0..2 {
        let hits = write_hits(&q, ctx, cols[i]);
        for col in 0..4 {
            let hit = hits[col];
            for row in 0..4 { a[row * 4 + col] = q.gate.select(ctx, zero, a[row * 4 + col], hit); }
        }
    }
    out.ints(&a);
    Ok(())
}
