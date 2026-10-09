// DS-1000 case 0353, literal translation of kernel.py (LITERAL_BASELINES.md).
// Axis maxima scan independently, followed by elementwise maximum and addition.
// Ravel uses row-major strides, with coordinate and dimension-product validity.
// in1d scans equality tests; masking uses running counts and equality-test compaction.
use crate::prelude::*;

fn maximum(q: &Q32, ctx: &mut Ctx, a: Cell, b: Cell) -> Cell {
    let a = q.bound(ctx, a);
    let b = q.bound(ctx, b);
    q.max(ctx, a, b).cell
}
fn ravel(q: &Q32, ctx: &mut Ctx, rows: &[Cell], dims: &[Cell]) -> Vec<Cell> {
    // NumPy rejects dimensions whose product does not fit a signed index.
    let mut size = q.int_const(ctx, 1);
    for d in dims {
        size = q.gate.mul(ctx, size, *d);
        q.range.range_check(ctx, size, 63);
    }
    let mut strides = vec![q.int_const(ctx, 1); 3];
    for k in (0..2).rev() {
        strides[k] = q.gate.mul(ctx, strides[k + 1], dims[k + 1]);
    }
    let mut result = Vec::new();
    for row in rows.chunks(3) {
        let mut flat = q.int_const(ctx, 0);
        for k in 0..3 {
            // Nonnegative coordinate and coordinate < dimension.
            q.range.range_check(ctx, row[k], 63);
            let x = q.bound(ctx, row[k]);
            let d = q.bound(ctx, dims[k]);
            let valid = q.lt(ctx, x, d);
            q.gate.assert_is_const(ctx, &valid, &Fr::ONE);
            let term = q.gate.mul(ctx, row[k], strides[k]);
            flat = q.add(ctx, flat, term);
        }
        result.push(flat);
    }
    result
}
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let av = inputs.ints("A", &[2, 3])?;
    let bv = inputs.ints("B", &[7, 3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_ints(ctx, &av);
    let b = q.load_ints(ctx, &bv);
    // B.max(0), then A.max(0).
    let mut bmax = Vec::new();
    for c in 0..3 {
        let mut acc = b[c];
        for r in 1..7 {
            acc = maximum(&q, ctx, acc, b[r * 3 + c]);
        }
        bmax.push(acc);
    }
    let mut amax = Vec::new();
    for c in 0..3 {
        let mut acc = a[c];
        for r in 1..2 {
            acc = maximum(&q, ctx, acc, a[r * 3 + c]);
        }
        amax.push(acc);
    }
    let mut maxima = Vec::new();
    for c in 0..3 {
        maxima.push(maximum(&q, ctx, bmax[c], amax[c]));
    }
    let dims: Vec<Cell> = maxima
        .iter()
        .map(|v| q.add(ctx, *v, Constant(Fr::ONE)))
        .collect();
    let flat_a = ravel(&q, ctx, &a, &dims);
    let flat_b = ravel(&q, ctx, &b, &dims);
    let mut member = Vec::new();
    for x in flat_a {
        let mut found = q.int_const(ctx, 0);
        for y in &flat_b {
            let eq = q.eq(ctx, x, *y);
            found = q.gate.or(ctx, found, eq);
        }
        member.push(found);
    }
    let keep: Vec<Cell> = member.iter().map(|x| q.gate.not(ctx, *x)).collect();
    let mut count = q.int_const(ctx, 0);
    let mut positions = Vec::new();
    for bit in &keep {
        positions.push(count);
        count = q.add(ctx, count, *bit);
    }
    let mut result = Vec::new();
    for j in 0..2 {
        let mut hits = Vec::new();
        for i in 0..2 {
            let eq = q.eq(ctx, positions[i], Constant(fe(j)));
            hits.push(q.gate.and(ctx, keep[i], eq));
        }
        for c in 0..3 {
            let mut value = q.int_const(ctx, 0);
            for i in 0..2 {
                let term = q.gate.mul(ctx, hits[i], a[i * 3 + c]);
                value = q.add(ctx, value, term);
            }
            result.push(value);
        }
    }
    out.dynamic(count, &result, Kind::Int, 3);
    Ok(())
}
