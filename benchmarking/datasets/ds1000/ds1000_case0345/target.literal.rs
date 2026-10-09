// DS-1000 case 0345, literal translation of kernel.py (LITERAL_BASELINES.md).
// Gather the augmented-assignment slice, compute each element, then scatter by equality selects.
// Negative indices wrap; each source indexed expression is evaluated separately.
// Reduction is a running accumulator in index order.
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
    let values: Vec<Cell> = (0..ROWS)
        .map(|i| q.index(ctx, &a[i * COLS..(i + 1) * COLS], idx1))
        .collect();
    let scaled: Vec<Cell> = values
        .iter()
        .map(|v| q.mul(ctx, *v, multiply_number).cell)
        .collect();
    for i in 0..ROWS {
        for k in 0..COLS {
            a[i * COLS + k] = q.gate.select(ctx, scaled[i], a[i * COLS + k], hits1[k]);
        }
    }

    // result = np.cumsum(a[:, col-1])
    let idx2 = q.gate.sub(ctx, col, Constant(Fr::ONE)); // recomputed: no CSE across statements
    let values: Vec<Cell> = (0..ROWS)
        .map(|i| q.index(ctx, &a[i * COLS..(i + 1) * COLS], idx2))
        .collect();
    let mut cumsum = vec![values[0]];
    for value in &values[1..] {
        cumsum.push(q.add(ctx, *cumsum.last().unwrap(), *value));
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
