// pythran_ising__ising_step: literal translation of kernel.py (LITERAL_BASELINES.md).
// L4: zero the inactive exponential argument before its range check.
// Evaluate both branches and select per update; float32 remains Q32; clamp and update mirror the source helpers.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = field_0
const M: usize = 2; // = field_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let field = inputs.ints("field", &[N, M])?;
    let beta = inputs.float("beta")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let mut field = q.load_ints(ctx, &field);
    let beta = q.load_float(ctx, beta);
    for n_offset in 0..2 {
        for m_offset in 0..2 {
            for n in (n_offset..N).step_by(2) {
                for m in (m_offset..M).step_by(2) {
                    update(&q, ctx, &mut field, beta, n, m);
                }
            }
        }
    }
    out.ints(&field);
    Ok(())
}

fn clamp(v: isize, n: usize) -> usize {
    if v < 0 {
        (v + n as isize) as usize
    } else if v < n as isize {
        v as usize
    } else {
        (v - n as isize) as usize
    }
}
fn update(q: &Q32, ctx: &mut Ctx, field: &mut [Cell], beta: Cell, n: usize, m: usize) {
    let mut total = ctx.load_zero();
    for i in n as isize - 1..n as isize + 2 {
        for j in m as isize - 1..m as isize + 2 {
            if i == n as isize && j == m as isize {
                continue;
            }
            total = q.add(ctx, total, field[clamp(i, N) * M + clamp(j, M)]);
        }
    }
    let twice = q.gate.mul(ctx, Constant(fe(2)), field[n * M + m]);
    let de = q.gate.mul(ctx, twice, total);
    let left = q.bound(ctx, de);
    let zero = q.bounded_const(ctx, 0);
    let first = q.le(ctx, left, zero);
    let first_value = q.gate.mul(ctx, field[n * M + m], Constant(fe(-1)));
    let negative = q.neg(ctx, de);
    let argument = q.mul_int(ctx, beta, negative);
    let argument = q.gate.mul_not(ctx, first, argument);
    let argument = q.bound(ctx, argument);
    let exponential = q.canonical_exp(ctx, argument).cell;
    let left = q.bound(ctx, exponential);
    let half = q.bounded_const(ctx, SCALE / 2);
    let second = q.gt(ctx, left, half);
    let second_value = q.gate.mul(ctx, field[n * M + m], Constant(fe(-1)));
    let otherwise = q.gate.select(ctx, second_value, field[n * M + m], second);
    field[n * M + m] = q.gate.select(ctx, first_value, otherwise, first);
}
