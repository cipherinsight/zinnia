//! DS-1000 case 0433: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Normalize each NumPy index with wrap/bounds, then scan positions and select; complete row writes before column writes.
use crate::prelude::*;

fn position(q: &Q32, ctx: &mut Ctx, idx: Cell) -> Cell {
    let ib = q.bound(ctx, idx);
    let negative = q.is_neg(ctx, ib);
    let wrapped = q.add(ctx, idx, Constant(fe(4)));
    let pos = q.gate.select(ctx, wrapped, idx, negative);
    q.range.check_less_than_safe(ctx, pos, 4);
    pos
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("a", &[4, 4])?;
    let rows = vec![inputs.int("zero_rows")?];
    let cols = vec![inputs.int("zero_cols")?];
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut a = q.load_ints(ctx, &av);
    let rows = q.load_ints(ctx, &rows);
    let cols = q.load_ints(ctx, &cols);
    let zero = q.int_const(ctx, 0);
    // a[zero_rows, :] = 0
    for i in 0..1 {
        let pos = position(&q, ctx, rows[i]);
        for row in 0..4 {
            let hit = q.eq(ctx, pos, Constant(fe(row as i128)));
            for col in 0..4 { a[row * 4 + col] = q.gate.select(ctx, zero, a[row * 4 + col], hit); }
        }
    }
    // a[:, zero_cols] = 0
    for i in 0..1 {
        let pos = position(&q, ctx, cols[i]);
        for col in 0..4 {
            let hit = q.eq(ctx, pos, Constant(fe(col as i128)));
            for row in 0..4 { a[row * 4 + col] = q.gate.select(ctx, zero, a[row * 4 + col], hit); }
        }
    }
    out.ints(&a);
    Ok(())
}
