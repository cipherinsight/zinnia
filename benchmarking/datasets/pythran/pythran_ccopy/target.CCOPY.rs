//! pythran_ccopy__CCOPY: bounded source loops and sequential strided copies.
//! Full 1024-step precondition bound, including zero and negative strides.
//! Inactive accesses use index zero; active accesses enforce NumPy bounds.
//! Reuse input bounds and complex-gather one-hots, with constant unit-stride indices.
use crate::prelude::*;
const NX: usize = 4; // = CX_0
const NY: usize = 4; // = CY_0
const PARTS: usize = 2; // complex real/imaginary components, not an input dimension
const MAX_STEPS: usize = 1024; // source precondition
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("N")?;
    let cx = inputs.complexes("CX", &[NX])?;
    let incx = inputs.int("INCX")?;
    let cy = inputs.complexes("CY", &[NY])?;
    let incy = inputs.int("INCY")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let n = q.load_int_bounded(ctx, n);
    let cx = q.load_floats(ctx, &cx);
    let incx = q.load_int_bounded(ctx, incx);
    let mut cy = q.load_floats(ctx, &cy);
    let incy = q.load_int_bounded(ctx, incy);
    let zero = q.int_const(ctx, 0);
    let negative = q.lt_const(ctx, n, 0);
    q.gate.assert_is_const(ctx, &negative, &Fr::ZERO);
    let too_large = q.gt_const(ctx, n, MAX_STEPS as i128);
    q.gate.assert_is_const(ctx, &too_large, &Fr::ZERO);
    let x_unit = q.eq(ctx, incx.cell, Constant(Fr::ONE));
    let y_unit = q.eq(ctx, incy.cell, Constant(Fr::ONE));
    let unit = q.gate.and(ctx, x_unit, y_unit);
    let strided = q.gate.not(ctx, unit);
    // N <= 0 returns unchanged CY; step < N masks every effect below.
    for step in 0..MAX_STEPS {
        let running = q.gt_const(ctx, n, step as i128);
        let active = q.gate.and(ctx, running, unit);
        // Constant propagation of the unit-stride source indices, not a loop-bound change.
        if step < NX && step < NY {
            for part in 0..PARTS {
                cy[PARTS * step + part] = q.gate.select(ctx, cx[PARTS * step + part], cy[PARTS * step + part], active);
            }
        } else {
            q.gate.assert_is_const(ctx, &active, &Fr::ZERO);
        }
    }
    let x_negative = q.is_neg(ctx, incx);
    let y_negative = q.is_neg(ctx, incy);
    let one_minus_n = q.sub(ctx, Constant(Fr::ONE), n.cell);
    let x_start = q.gate.mul(ctx, one_minus_n, incx.cell);
    let y_start = q.gate.mul(ctx, one_minus_n, incy.cell);
    let mut ix = q.gate.select(ctx, x_start, zero, x_negative);
    let mut iy = q.gate.select(ctx, y_start, zero, y_negative);
    for step in 0..MAX_STEPS {
        let running = q.gt_const(ctx, n, step as i128);
        let active = q.gate.and(ctx, running, strided);
        let safe_x = q.gate.select(ctx, ix, zero, active);
        let safe_y = q.gate.select(ctx, iy, zero, active);
        let x_hits = q.index_hits(ctx, safe_x, NX);
        let y_hits = q.index_hits(ctx, safe_y, NY);
        let value: Vec<Cell> = (0..PARTS).map(|part| {
            q.gate.inner_product(ctx, x_hits.iter().copied().map(Existing),
                (0..NX).map(|i| Existing(cx[PARTS * i + part])))
        }).collect();
        for j in 0..NY {
            let write = q.gate.and(ctx, active, y_hits[j]);
            for part in 0..PARTS {
                cy[PARTS * j + part] = q.gate.select(ctx, value[part], cy[PARTS * j + part], write);
            }
        }
        let next_x = q.add(ctx, ix, incx.cell);
        let next_y = q.add(ctx, iy, incy.cell);
        ix = q.gate.select(ctx, next_x, ix, active);
        iy = q.gate.select(ctx, next_y, iy, active);
    }
    out.floats(&cy);
    Ok(())
}
