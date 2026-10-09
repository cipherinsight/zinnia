// DS-1000 case 0346, literal translation of kernel.py (LITERAL_BASELINES.md).
// Gather the augmented-assignment slice, compute each element, then scatter by equality selects.
// Negative indices wrap; each source indexed expression is evaluated separately.
// Reduction is a running accumulator in index order.
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
    let values: Vec<Cell> = (0..COLS)
        .map(|j| {
            let column: Vec<Cell> = (0..ROWS).map(|i| a[i * COLS + j]).collect();
            q.index(ctx, &column, idx1)
        })
        .collect();
    let scaled: Vec<Cell> = values
        .iter()
        .map(|v| q.mul(ctx, *v, multiply_number).cell)
        .collect();
    for i in 0..ROWS {
        for j in 0..COLS {
            a[i * COLS + j] = q.gate.select(ctx, scaled[j], a[i * COLS + j], hits1[i]);
        }
    }

    // result = np.cumsum(a[row-1, :])
    let idx2 = q.gate.sub(ctx, row, Constant(Fr::ONE)); // recomputed: no CSE across statements
    let values: Vec<Cell> = (0..COLS)
        .map(|j| {
            let column: Vec<Cell> = (0..ROWS).map(|i| a[i * COLS + j]).collect();
            q.index(ctx, &column, idx2)
        })
        .collect();
    let mut cumsum = vec![values[0]];
    for value in &values[1..] {
        cumsum.push(q.add(ctx, *cumsum.last().unwrap(), *value));
    }
    out.floats(&cumsum);
    Ok(())
}

/// A position matches its nonnegative index or its wrapped negative index.
/// Exactly one match is required, rejecting all out-of-range indices.
fn row_hits(q: &Q32, ctx: &mut Ctx, idx: Cell, len: usize) -> Vec<Cell> {
    let hits: Vec<Cell> = (0..len)
        .map(|k| {
            let positive = q.eq(ctx, idx, Constant(fe(k as i128)));
            let negative = q.eq(ctx, idx, Constant(fe(k as i128 - len as i128)));
            q.gate.or(ctx, positive, negative)
        })
        .collect();
    let total = q.sum(ctx, hits.iter().copied());
    q.gate.assert_is_const(ctx, &total, &Fr::ONE);
    hits
}
