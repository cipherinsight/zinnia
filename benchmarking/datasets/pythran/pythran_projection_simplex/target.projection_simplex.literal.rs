//! pythran_projection_simplex__projection_simplex: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Stable bubble sort, reverse, prefix sum. Both boolean-index calls independently compact
//! by prefix counts and pairwise equality tests. A nonempty check precedes each [-1] read.
use crate::prelude::*;
const N: usize = 10; // = v_0
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("v", &[N])?;
    let z = inputs.int("z")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let v = q.load_floats(ctx, &raw);
    let z = q.load_int(ctx, z);
    let mut sorted = v.clone();
    for pass in 0..N - 1 {
        for j in 0..N - 1 - pass {
            let left = q.bound(ctx, sorted[j]);
            let right = q.bound(ctx, sorted[j + 1]);
            let swap = q.gt(ctx, left, right);
            let (a, b) = (sorted[j], sorted[j + 1]);
            sorted[j] = q.gate.select(ctx, b, a, swap);
            sorted[j + 1] = q.gate.select(ctx, a, b, swap);
        }
    }
    let u: Vec<Cell> = sorted.into_iter().rev().collect();
    let mut sum = q.float_const(ctx, 0.0);
    let mut cumsum = Vec::new();
    for x in &u {
        sum = q.add(ctx, sum, *x);
        cumsum.push(sum);
    }
    let zf = q.int_to_float(ctx, z);
    let cssv: Vec<Cell> = cumsum.iter().map(|v| q.sub(ctx, *v, zf)).collect();
    let ind: Vec<Cell> = (0..N).map(|i| q.int_const(ctx, (i + 1) as i64)).collect();
    let quotients: Vec<Cell> = (0..N)
        .map(|i| q.floor_div_const(ctx, cssv[i], (i + 1) as u64).0.cell)
        .collect();
    let gaps: Vec<Cell> = (0..N).map(|i| q.sub(ctx, u[i], quotients[i])).collect();
    let cond: Vec<Cell> = gaps
        .iter()
        .map(|v| {
            let a = q.bound(ctx, *v);
            let zero = q.bounded_const(ctx, 0);
            q.gt(ctx, a, zero)
        })
        .collect();
    let (len, selected) = compact_literal(&q, ctx, &ind, &cond);
    let empty = q.eq(ctx, len, Constant(Fr::ZERO));
    q.gate.assert_is_const(ctx, &empty, &Fr::ZERO);
    let last = q.sub(ctx, len, Constant(Fr::ONE));
    let rho = q.index(ctx, &selected, last);
    let (len, selected) = compact_literal(&q, ctx, &cssv, &cond);
    let empty = q.eq(ctx, len, Constant(Fr::ZERO));
    q.gate.assert_is_const(ctx, &empty, &Fr::ZERO);
    let last = q.sub(ctx, len, Constant(Fr::ONE));
    let numerator = q.index(ctx, &selected, last);
    let rho_float = q.int_to_float(ctx, rho);
    let theta = q.canonical_div_nonzero(ctx, numerator, rho_float).cell;
    let differences: Vec<Cell> = v.iter().map(|v| q.sub(ctx, *v, theta)).collect();
    let w: Vec<Cell> = differences
        .iter()
        .map(|v| {
            let v = q.bound(ctx, *v);
            let zero = q.bounded_const(ctx, 0);
            q.max(ctx, v, zero).cell
        })
        .collect();
    out.floats(&w);
    Ok(())
}
fn compact_literal(q: &Q32, ctx: &mut Ctx, xs: &[Cell], keep: &[Cell]) -> (Cell, Vec<Cell>) {
    let mut count = q.int_const(ctx, 0);
    let mut positions = Vec::new();
    for k in keep {
        positions.push(count);
        count = q.add(ctx, count, *k);
    }
    let mut result = Vec::new();
    for j in 0..N {
        let mut value = q.int_const(ctx, 0);
        for i in 0..N {
            let at = q.eq(ctx, positions[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], at);
            let term = q.gate.mul(ctx, hit, xs[i]);
            value = q.add(ctx, value, term);
        }
        result.push(value);
    }
    (count, result)
}
