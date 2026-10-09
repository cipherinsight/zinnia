// pythran_periodic_dist__dist: literal translation of kernel.py (LITERAL_BASELINES.md).
// L4: inactive axis updates use L=0 so L/2 cannot overflow. Radicand guard matches optimized.
// Tile/transpose are wires. Mask reads use L5 equality-test compaction; masked writes scatter by count equality. Private flags select complete updated arrays; no symmetry.
use crate::prelude::*;
const N: usize = 16; // = x_0
const PAIRS: usize = 256; // = x_0 * x_0
                          // Implements one source masked assignment, RHS before LHS.
fn update(q: &Q32, ctx: &mut Ctx, values: &[Cell], l: Cell, negative: bool) -> Vec<Cell> {
    let signed_l = if negative { q.neg(ctx, l) } else { l };
    let lf = q.int_to_float(ctx, signed_l);
    let half = q.div_const(ctx, lf, 2.0).cell;
    let mut keep = Vec::with_capacity(PAIRS);
    for value in values {
        let a = q.bound(ctx, *value);
        let b = q.bound(ctx, half);
        keep.push(if negative {
            q.lt(ctx, a, b)
        } else {
            q.gt(ctx, a, b)
        });
    }
    let zero = q.int_const(ctx, 0);
    let mut count = zero;
    let mut positions = Vec::with_capacity(PAIRS);
    for kept in &keep {
        positions.push(count);
        count = q.add(ctx, count, *kept);
    }
    let mut compacted = vec![zero; PAIRS];
    for j in 0..PAIRS {
        let mut sum = zero;
        for i in 0..PAIRS {
            let eq = q.eq(ctx, positions[i], Constant(fe(j as i128)));
            let hit = q.gate.and(ctx, keep[i], eq);
            let term = q.gate.mul(ctx, hit, values[i]);
            sum = q.add(ctx, sum, term);
        }
        compacted[j] = sum;
    }
    let lf = q.int_to_float(ctx, l);
    let changed: Vec<Cell> = compacted
        .iter()
        .map(|v| {
            if negative {
                q.add(ctx, *v, lf)
            } else {
                q.sub(ctx, *v, lf)
            }
        })
        .collect();
    // Target mask is a distinct source expression and is evaluated again.
    let signed_l = if negative { q.neg(ctx, l) } else { l };
    let lf = q.int_to_float(ctx, signed_l);
    let half = q.div_const(ctx, lf, 2.0).cell;
    let mut target_keep = Vec::with_capacity(PAIRS);
    for value in values {
        let a = q.bound(ctx, *value);
        let b = q.bound(ctx, half);
        target_keep.push(if negative {
            q.lt(ctx, a, b)
        } else {
            q.gt(ctx, a, b)
        });
    }
    let mut next = zero;
    let mut target_positions = Vec::with_capacity(PAIRS);
    for kept in &target_keep {
        target_positions.push(next);
        next = q.add(ctx, next, *kept);
    }
    let mut result = values.to_vec();
    for i in 0..PAIRS {
        let mut replacement = zero;
        for j in 0..PAIRS {
            let hit = q.eq(ctx, target_positions[i], Constant(fe(j as i128)));
            let term = q.gate.mul(ctx, hit, changed[j]);
            replacement = q.add(ctx, replacement, term);
        }
        result[i] = q.gate.select(ctx, replacement, values[i], target_keep[i]);
    }
    result
}

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N])?;
    let y = inputs.floats("y", &[N])?;
    let z = inputs.floats("z", &[N])?;
    let l = inputs.int("L")?;
    let px = inputs.bools("periodicX", &[])?;
    let py = inputs.bools("periodicY", &[])?;
    let pz = inputs.bools("periodicZ", &[])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let y = q.load_floats(ctx, &y);
    let z = q.load_floats(ctx, &z);
    let l = q.load_int(ctx, l);
    let px = q.load_bools(ctx, &px)[0];
    let py = q.load_bools(ctx, &py)[0];
    let pz = q.load_bools(ctx, &pz)[0];
    let zero = q.float_const(ctx, 0.0);
    let xtemp: Vec<Cell> = (0..N).flat_map(|_| x.iter().copied()).collect();
    let mut dx: Vec<Cell> = (0..PAIRS)
        .map(|at| q.sub(ctx, xtemp[at], xtemp[(at % N) * N + at / N]))
        .collect();
    let ytemp: Vec<Cell> = (0..N).flat_map(|_| y.iter().copied()).collect();
    let mut dy: Vec<Cell> = (0..PAIRS)
        .map(|at| q.sub(ctx, ytemp[at], ytemp[(at % N) * N + at / N]))
        .collect();
    let ztemp: Vec<Cell> = (0..N).flat_map(|_| z.iter().copied()).collect();
    let mut dz: Vec<Cell> = (0..PAIRS)
        .map(|at| q.sub(ctx, ztemp[at], ztemp[(at % N) * N + at / N]))
        .collect();
    let safe_l = q.gate.mul(ctx, l, px);
    let first = update(&q, ctx, &dx, safe_l, false);
    let second = update(&q, ctx, &first, safe_l, true);
    dx = (0..PAIRS)
        .map(|at| q.gate.select(ctx, second[at], dx[at], px))
        .collect();
    let safe_l = q.gate.mul(ctx, l, py);
    let first = update(&q, ctx, &dy, safe_l, false);
    let second = update(&q, ctx, &first, safe_l, true);
    dy = (0..PAIRS)
        .map(|at| q.gate.select(ctx, second[at], dy[at], py))
        .collect();
    let safe_l = q.gate.mul(ctx, l, pz);
    let first = update(&q, ctx, &dz, safe_l, false);
    let second = update(&q, ctx, &first, safe_l, true);
    dz = (0..PAIRS)
        .map(|at| q.gate.select(ctx, second[at], dz[at], pz))
        .collect();
    let sx: Vec<Cell> = (0..PAIRS).map(|at| q.pow_const(ctx, dx[at], 2)).collect();
    let sy: Vec<Cell> = (0..PAIRS).map(|at| q.pow_const(ctx, dy[at], 2)).collect();
    let xy: Vec<Cell> = (0..PAIRS).map(|at| q.add(ctx, sx[at], sy[at])).collect();
    let sz: Vec<Cell> = (0..PAIRS).map(|at| q.pow_const(ctx, dz[at], 2)).collect();
    let xyz: Vec<Cell> = (0..PAIRS).map(|at| q.add(ctx, xy[at], sz[at])).collect();
    let roots: Vec<Cell> = (0..PAIRS).map(|at| { let sum = q.bound(ctx, xyz[at]).cell; q.sqrt(ctx, sum).cell }).collect();
    let mask: Vec<Cell> = roots.iter().map(|v| q.eq(ctx, *v, zero)).collect();
    let d: Vec<Cell> = (0..PAIRS)
        .map(|at| {
            q.gate
                .select(ctx, Constant(fe(-SCALE)), roots[at], mask[at])
        })
        .collect();
    out.floats(&d);
    out.floats(&dx);
    out.floats(&dy);
    out.floats(&dz);
    Ok(())
}
