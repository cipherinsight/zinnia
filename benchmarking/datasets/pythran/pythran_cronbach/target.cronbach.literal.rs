//! pythran_cronbach__cronbach: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Variance uses mean, full deviations, rounded squares, sum, then ddof=1 division.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const ROWS: usize = 16; // = itemscores_0
const COLS: usize = 16; // = itemscores_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let itemscores = inputs.floats("itemscores", &[ROWS, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let scores = q.load_floats(ctx, &itemscores);
    // itemvars = itemscores.var(1, None, None, 1)
    let mut means = Vec::new();
    for row in scores.chunks(COLS) { let sum = sum(&q, ctx, row); means.push(q.floor_div_const(ctx, sum, COLS as u64).0.cell); }
    let deviations: Vec<Cell> = (0..ROWS*COLS).map(|i| q.sub(ctx, scores[i], means[i/COLS])).collect();
    let squares: Vec<Cell> = deviations.iter().map(|v| q.mul(ctx, *v, *v).cell).collect();
    let mut itemvars = Vec::new();
    for row in squares.chunks(COLS) { let total = sum(&q, ctx, row); itemvars.push(q.floor_div_const(ctx, total, (COLS-1) as u64).0.cell); }
    let mut tscores = Vec::new();
    for j in 0..COLS {
        let column: Vec<Cell> = (0..ROWS).map(|i| scores[i*COLS+j]).collect();
        tscores.push(sum(&q, ctx, &column));
    }
    let itemvars_sum = sum(&q, ctx, &itemvars);
    let total = sum(&q, ctx, &tscores);
    let mean = q.floor_div_const(ctx, total, COLS as u64).0.cell;
    let deviations: Vec<Cell> = tscores.iter().map(|v| q.sub(ctx, *v, mean)).collect();
    let squares: Vec<Cell> = deviations.iter().map(|v| q.mul(ctx, *v, *v).cell).collect();
    let total = sum(&q, ctx, &squares);
    let tvar = q.floor_div_const(ctx, total, (COLS-1) as u64).0.cell;
    let ratio = q.canonical_div_nonzero(ctx, itemvars_sum, tvar).cell;
    let complement = q.sub(ctx, Constant(fe(SCALE)), ratio);
    let result = q.mul(ctx, Constant(fe(quantize(ROWS as f64/(ROWS-1) as f64))), complement).cell;
    out.float(result); Ok(())
}
fn sum(q: &Q32, ctx: &mut Ctx, xs: &[Cell]) -> Cell {
    let mut acc = q.float_const(ctx, 0.0);
    for v in xs { acc = q.add(ctx, acc, *v); } acc
}
