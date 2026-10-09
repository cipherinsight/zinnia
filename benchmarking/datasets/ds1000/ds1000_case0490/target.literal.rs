//! DS-1000 case 0490: literal translation of kernel.py (LITERAL_BASELINES.md).
//! where and boolean indexing use count/equality compaction; writes scan every position.
//! Private-length loops run 50 slots with active flags; padding reads index zero safely.
//! Mixed comparisons convert each integer threshold at the corresponding element operation.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let raw = inputs.floats("arr", &[5, 50])?;
    let r1 = inputs.ints("n1", &[5])?;
    let r2 = inputs.ints("n2", &[5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let arr = q.load_floats(ctx, &raw);
    let n1 = q.load_ints(ctx, &r1);
    let n2 = q.load_ints(ctx, &r2);
    let mut result = Vec::new();
    for r in 0..5 {
        let mut a = arr[r * 50..(r + 1) * 50].to_vec();
        let temp = a.clone(); // temp = a.copy()
        let mut mask = Vec::new();
        for i in 0..50 {
            let t = q.int_to_float(ctx, n1[r]);
            let lhs = q.bound(ctx, temp[i]);
            let rhs = q.bound(ctx, t);
            mask.push(q.lt(ctx, lhs, rhs));
        }
        let (count, pos) = positions(&q, ctx, &mask); // np.where(temp < t1)
        for j in 0..50 {
            let jc = q_const(ctx, j);
            let jb = q.bound(ctx, jc);
            let cb = q.bound(ctx, count);
            let active = q.lt(ctx, jb, cb);
            for k in 0..50 {
                let eq = q.eq(ctx, pos[j], Constant(fe(k as i128)));
                let hit = q.gate.and(ctx, active, eq);
                a[k] = q.gate.select(ctx, Constant(Fr::ZERO), a[k], hit);
            }
        }
        let mut mask = Vec::new();
        for i in 0..50 {
            let t = q.int_to_float(ctx, n2[r]);
            let lhs = q.bound(ctx, temp[i]);
            let rhs = q.bound(ctx, t);
            mask.push(q.ge(ctx, lhs, rhs));
        }
        let (count, pos) = positions(&q, ctx, &mask); // np.where(temp >= t2)
        for j in 0..50 {
            let jc = q_const(ctx, j);
            let jb = q.bound(ctx, jc);
            let cb = q.bound(ctx, count);
            let active = q.lt(ctx, jb, cb);
            for k in 0..50 {
                let eq = q.eq(ctx, pos[j], Constant(fe(k as i128)));
                let hit = q.gate.and(ctx, active, eq);
                a[k] = q.gate.select(ctx, Constant(fe(30 * SCALE)), a[k], hit);
            }
        }
        let mut lower = Vec::new();
        for i in 0..50 {
            let t = q.int_to_float(ctx, n1[r]);
            let lhs = q.bound(ctx, temp[i]);
            let rhs = q.bound(ctx, t);
            lower.push(q.ge(ctx, lhs, rhs));
        }
        let mut upper = Vec::new();
        for i in 0..50 {
            let t = q.int_to_float(ctx, n2[r]);
            let lhs = q.bound(ctx, temp[i]);
            let rhs = q.bound(ctx, t);
            upper.push(q.lt(ctx, lhs, rhs));
        }
        let mask: Vec<Cell> = (0..50)
            .map(|i| q.gate.and(ctx, lower[i], upper[i]))
            .collect();
        // Materialize the masked read with L5 compaction, before adding five.
        let (count, selected) = gather_mask(&q, ctx, &a, &mask);
        let values: Vec<Cell> = selected
            .iter()
            .map(|v| q.add(ctx, *v, Constant(fe(5 * SCALE))))
            .collect();
        // Scatter the modified compact array to the selected source positions.
        let mut before = ctx.load_zero();
        for k in 0..50 {
            for j in 0..50 {
                let eq = q.eq(ctx, before, Constant(fe(j as i128)));
                let hit = q.gate.and(ctx, mask[k], eq);
                a[k] = q.gate.select(ctx, values[j], a[k], hit);
            }
            before = q.add(ctx, before, mask[k]);
        }
        let _ = count;
        result.extend(a);
    }
    out.floats(&result);
    Ok(())
}
fn q_const(ctx: &mut Ctx, i: usize) -> Cell {
    ctx.load_constant(fe(i as i128))
}
fn positions(q: &Q32, ctx: &mut Ctx, mask: &[Cell]) -> (Cell, Vec<Cell>) {
    let payload: Vec<Cell> = (0..50).map(|i| q_const(ctx, i)).collect();
    gather_mask(q, ctx, &payload, mask)
}
fn gather_mask(q: &Q32, ctx: &mut Ctx, payload: &[Cell], mask: &[Cell]) -> (Cell, Vec<Cell>) {
    let mut count = ctx.load_zero();
    let mut before = Vec::new();
    for i in 0..50 {
        before.push(count);
        count = q.add(ctx, count, mask[i]);
    }
    let mut result = Vec::new();
    for j in 0..50 {
        let mut acc = ctx.load_zero();
        for i in 0..50 {
            let eq = q.eq(ctx, before[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, mask[i], eq);
            let term = q.gate.mul(ctx, hit, payload[i]);
            acc = q.add(ctx, acc, term);
        }
        result.push(acc);
    }
    (count, result)
}
