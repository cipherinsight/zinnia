//! DS-1000 case 0509: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Coordinate arrays use prefix-count/equality compaction, then independent min/max scans.
//! Dynamic loops run to the input shape, with active flags; slices use standard indexed reads.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("im", &[10, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let image = q.load_floats(ctx, &raw);
    let mask: Vec<Cell> = image
        .iter()
        .map(|v| q.eq(ctx, *v, Constant(Fr::ZERO)))
        .collect();
    let mut sums = Vec::new();
    for i in 0..10 {
        let mut sum = ctx.load_zero();
        for j in 0..10 {
            sum = q.add(ctx, sum, mask[i * 10 + j]);
        }
        sums.push(sum);
    }
    let keep: Vec<Cell> = sums
        .iter()
        .map(|v| {
            let z = q.eq(ctx, *v, Constant(Fr::ZERO));
            q.gate.not(ctx, z)
        })
        .collect();
    let payload: Vec<Vec<Cell>> = (0..10).map(|i| vec![q.int_const(ctx, i as i64)]).collect();
    let (rows_count, rows) = nonzero(&q, ctx, &keep, &payload);
    let rows: Vec<Cell> = rows.iter().map(|v| v[0]).collect();
    let mut sums = Vec::new();
    for i in 0..10 {
        let mut sum = ctx.load_zero();
        for j in 0..10 {
            sum = q.add(ctx, sum, mask[j * 10 + i]);
        }
        sums.push(sum);
    }
    let keep: Vec<Cell> = sums
        .iter()
        .map(|v| {
            let z = q.eq(ctx, *v, Constant(Fr::ZERO));
            q.gate.not(ctx, z)
        })
        .collect();
    let payload: Vec<Vec<Cell>> = (0..10).map(|i| vec![q.int_const(ctx, i as i64)]).collect();
    let (cols_count, cols) = nonzero(&q, ctx, &keep, &payload);
    let cols: Vec<Cell> = cols.iter().map(|v| v[0]).collect();
    let empty = q.eq(ctx, rows_count, Constant(Fr::ZERO));
    let present = q.gate.not(ctx, empty);
    // Both branches: inactive reductions have padded dummy inputs, with effects selected away.
    let top = reduce(&q, ctx, &rows, rows_count, false);
    let bottom = reduce(&q, ctx, &rows, rows_count, true);
    let bottom = q.add(ctx, bottom, Constant(Fr::ONE));
    let left = reduce(&q, ctx, &cols, cols_count, false);
    let right = reduce(&q, ctx, &cols, cols_count, true);
    let right = q.add(ctx, right, Constant(Fr::ONE));
    let (height, width, result) = crop(&q, ctx, &image, top, bottom, left, right, present, 10, 10);
    out.dynamic2(height, width, &result, Kind::Float, 10, 10);
    Ok(())
}

// L5 compaction of coordinate tuples: one equality test per item and slot.
fn nonzero(q: &Q32, ctx: &mut Ctx, keep: &[Cell], payload: &[Vec<Cell>]) -> (Cell, Vec<Vec<Cell>>) {
    let mut before = Vec::new();
    let mut count = ctx.load_zero();
    for k in keep {
        before.push(count);
        count = q.add(ctx, count, *k);
    }
    let mut result = Vec::new();
    for j in 0..keep.len() {
        let mut row = vec![ctx.load_zero(); payload[0].len()];
        for i in 0..keep.len() {
            let eq = q.eq(ctx, before[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], eq);
            for u in 0..row.len() {
                let term = q.gate.mul(ctx, hit, payload[i][u]);
                row[u] = q.add(ctx, row[u], term);
            }
        }
        result.push(row);
    }
    (count, result)
}
// A dynamic reduction runs its shape bound; padding cannot change the accumulator.
fn reduce(q: &Q32, ctx: &mut Ctx, xs: &[Cell], count: Cell, maximum: bool) -> Cell {
    let mut acc = xs[0];
    for i in 1..xs.len() {
        let ic = ctx.load_constant(fe(i as i128));
        let ib = q.bound(ctx, ic);
        let cb = q.bound(ctx, count);
        let active = q.lt(ctx, ib, cb);
        let lhs = q.bound(ctx, acc);
        let rhs = q.bound(ctx, xs[i]);
        let candidate = if maximum {
            q.max(ctx, lhs, rhs).cell
        } else {
            q.min(ctx, lhs, rhs).cell
        };
        acc = q.gate.select(ctx, candidate, acc, active);
    }
    acc
}
// A rectangular slice has private dimensions; enumerate its row-major output positions.
fn crop(
    q: &Q32,
    ctx: &mut Ctx,
    image: &[Cell],
    top: Cell,
    bottom: Cell,
    left: Cell,
    right: Cell,
    present: Cell,
    rows: usize,
    cols: usize,
) -> (Cell, Cell, Vec<Cell>) {
    let height = q.sub(ctx, bottom, top);
    let width = q.sub(ctx, right, left);
    let height = q.gate.select(ctx, height, Constant(Fr::ZERO), present);
    let width = q.gate.select(ctx, width, Constant(Fr::ZERO), present);
    let size = q.gate.mul(ctx, height, width);
    let mut values = Vec::new();
    for s in 0..rows * cols {
        let sc = ctx.load_constant(fe(s as i128));
        let sb = q.bound(ctx, sc);
        let sizeb = q.bound(ctx, size);
        let active = q.lt(ctx, sb, sizeb);
        // The inactive empty branch must not divide by zero or read out of range.
        let safe_width = q.gate.select(ctx, width, Constant(Fr::ONE), present);
        let wb = q.bound(ctx, safe_width);
        let (dy, dx) = q.floor_div_pos(ctx, sc, wb);
        let y = q.add(ctx, top, dy.cell);
        let x = q.add(ctx, left, dx);
        let row = q.gate.mul(ctx, y, Constant(fe(cols as i128)));
        let index = q.add(ctx, row, x);
        let safe_index = q.gate.select(ctx, index, Constant(Fr::ZERO), active);
        let value = q.index(ctx, image, safe_index);
        values.push(q.gate.select(ctx, value, Constant(Fr::ZERO), active));
    }
    (height, width, values)
}
