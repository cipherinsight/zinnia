// pythran_crotg__CROTG: literal translation of kernel.py (LITERAL_BASELINES.md).
// Repeated abs calls remain separate. Complex primitives retain current Q32 rounding. Inactive zero divisions are totalized and nonzero assertions guarded; outputs select both branches.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let ca = inputs.complex("CA")?;
    let cb = inputs.complex("CB")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let ca = q.load_floats(ctx, &ca);
    let cb = q.load_floats(ctx, &cb);
    let ca = [ca[0], ca[1]];
    let cb = [cb[0], cb[1]];
    let condition_abs = magnitude(&q, ctx, ca);
    let zero = q.eq(ctx, condition_abs, Constant(fe(0)));
    let active = q.gate.not(ctx, zero);
    // Else: each abs(CA) is a separate source occurrence.
    let a_abs = magnitude(&q, ctx, ca);
    let b_abs = magnitude(&q, ctx, cb);
    let scale = q.add(ctx, a_abs, b_abs);
    let a_ratio = complex_div(&q, ctx, ca, scale, active);
    let a_magnitude = magnitude(&q, ctx, a_ratio);
    let a_squared = q.pow_const(ctx, a_magnitude, 2);
    let b_ratio = complex_div(&q, ctx, cb, scale, active);
    let b_magnitude = magnitude(&q, ctx, b_ratio);
    let b_squared = q.pow_const(ctx, b_magnitude, 2);
    let sum = q.add(ctx, a_squared, b_squared);
    let root = q.sqrt(ctx, sum).cell;
    let norm = q.mul(ctx, scale, root).cell;
    let a_abs = magnitude(&q, ctx, ca);
    let alpha = complex_div(&q, ctx, ca, a_abs, active);
    let a_abs = magnitude(&q, ctx, ca);
    let c = divide(&q, ctx, a_abs, norm, active);
    let conjugate = [cb[0], q.neg(ctx, cb[1])];
    let arbr = q.mul(ctx, alpha[0], conjugate[0]).cell;
    let aibi = q.mul(ctx, alpha[1], conjugate[1]).cell;
    let arbi = q.mul(ctx, alpha[0], conjugate[1]).cell;
    let aibr = q.mul(ctx, alpha[1], conjugate[0]).cell;
    let product = [q.sub(ctx, arbr, aibi), q.add(ctx, arbi, aibr)];
    let s = complex_div(&q, ctx, product, norm, active);
    let new_ca = [
        q.mul(ctx, alpha[0], norm).cell,
        q.mul(ctx, alpha[1], norm).cell,
    ];
    let ca_result = [
        q.gate.select(ctx, cb[0], new_ca[0], zero),
        q.gate.select(ctx, cb[1], new_ca[1], zero),
    ];
    let c_result = q.gate.select(ctx, Constant(fe(0)), c, zero);
    let s_result = [
        q.gate.select(ctx, Constant(fe(SCALE)), s[0], zero),
        q.gate.select(ctx, Constant(fe(0)), s[1], zero),
    ];
    out.floats(&ca_result);
    out.floats(&cb);
    out.float(c_result);
    out.floats(&s_result);
    Ok(())
}
// Complex primitive schedule, matching the current shared arithmetic contract.
fn magnitude(q: &Q32, ctx: &mut Ctx, x: [Cell; 2]) -> Cell {
    let rr = q.mul(ctx, x[0], x[0]).cell;
    let ii = q.mul(ctx, x[1], x[1]).cell;
    let sum = q.add(ctx, rr, ii);
    let sum = q.bound(ctx, sum).cell;
    q.sqrt(ctx, sum).cell
}
// Totalize zero division only on the inactive branch, retaining its operands.
fn divide(q: &Q32, ctx: &mut Ctx, n: Cell, d: Cell, active: Cell) -> Cell {
    let zero = q.eq(ctx, d, Constant(fe(0)));
    let invalid = q.gate.and(ctx, active, zero);
    q.gate.assert_is_const(ctx, &invalid, &fe(0));
    let safe = q.add(ctx, d, zero);
    let numerator = q.gate.mul_not(ctx, zero, n);
    q.canonical_div_nonzero(ctx, numerator, safe).cell
}
fn complex_div(q: &Q32, ctx: &mut Ctx, x: [Cell; 2], d: Cell, active: Cell) -> [Cell; 2] {
    let denominator = q.mul(ctx, d, d).cell;
    let re = q.mul(ctx, x[0], d).cell;
    let im = q.mul(ctx, x[1], d).cell;
    [
        divide(q, ctx, re, denominator, active),
        divide(q, ctx, im, denominator, active),
    ]
}
