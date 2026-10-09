// DS-1000 case 0347, literal translation of kernel.py (LITERAL_BASELINES.md).
// Gather the augmented-assignment slice, compute each element, then scatter by equality selects.
// Negative indices wrap; each source indexed expression is evaluated separately.
// Reduction is a running accumulator in index order.
use crate::prelude::*;

const ROWS: usize = 8;
const COLS: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[ROWS, COLS])?;
    let row = inputs.int("row")?;
    let divide_number = inputs.float("divide_number")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut a = q.load_floats(ctx, &a);
    let row = q.load_int(ctx, row);
    let divisor = q.load_float(ctx, divide_number);

    // a[row-1, :] /= divide_number
    let idx1 = q.gate.sub(ctx, row, Constant(Fr::ONE));
    let hits1 = row_hits(&q, ctx, idx1, ROWS);
    let values: Vec<Cell> = (0..COLS).map(|j| {
        let column: Vec<Cell> = (0..ROWS).map(|i| a[i*COLS+j]).collect();
        q.index(ctx, &column, idx1)
    }).collect();
    let scaled: Vec<Cell> = values.iter().map(|v| { let d = q.bound(ctx, divisor); q.div(ctx, *v, d).cell }).collect();
    for i in 0..ROWS { for j in 0..COLS {
        a[i*COLS+j] = q.gate.select(ctx, scaled[j], a[i*COLS+j], hits1[i]);
    }}

    // result = np.multiply.reduce(a[row-1, :])
    let idx2 = q.gate.sub(ctx, row, Constant(Fr::ONE)); // recomputed: no CSE across statements
    let row_values: Vec<Cell> = (0..COLS)
        .map(|j| {
            let column: Vec<Cell> = (0..ROWS).map(|i| a[i * COLS + j]).collect();
            q.index(ctx, &column, idx2)
        })
        .collect();
    let product = row_values[1..]
        .iter()
        .fold(row_values[0], |acc, x| q.mul(ctx, acc, *x).cell);
    out.float(product);
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
