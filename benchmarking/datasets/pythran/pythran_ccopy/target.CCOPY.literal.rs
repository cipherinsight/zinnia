//! pythran_ccopy__CCOPY: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Each range(N) runs to the source precondition's 1024 bound with active flags.
//! Inactive branches/iterations use index zero to avoid spurious IndexError.
//! Private writes scan every CY position; each complex component uses q.index.
//! Index arithmetic is exact field arithmetic; only active array accesses are bounded.
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
    let n = q.load_int(ctx, n);
    let cx = q.load_floats(ctx, &cx);
    let incx = q.load_int(ctx, incx);
    let cy = q.load_floats(ctx, &cy);
    let incy = q.load_int(ctx, incy);
    let zero = q.int_const(ctx, 0);
    // @requires 0 <= N <= 1024
    let nb = q.bound(ctx, n);
    let negative = q.lt_const(ctx, nb, 0);
    q.gate.assert_is_const(ctx, &negative, &Fr::ZERO);
    let nb = q.bound(ctx, n);
    let too_large = q.gt_const(ctx, nb, MAX_STEPS as i128);
    q.gate.assert_is_const(ctx, &too_large, &Fr::ZERO);
    // if N <= 0: return
    let nb = q.bound(ctx, n);
    let positive = q.gt_const(ctx, nb, 0);
    // if INCX == 1 and INCY == 1
    let x_unit = q.eq(ctx, incx, Constant(Fr::ONE));
    let y_unit = q.eq(ctx, incy, Constant(Fr::ONE));
    let unit = q.gate.and(ctx, x_unit, y_unit);
    let fast = q.gate.and(ctx, positive, unit);
    let not_unit = q.gate.not(ctx, unit);
    let slow = q.gate.and(ctx, positive, not_unit);
    let mut fast_cy = cy.clone();
    for step in 0..MAX_STEPS {
        let nb = q.bound(ctx, n);
        let running = q.gt_const(ctx, nb, step as i128);
        let active = q.gate.and(ctx, fast, running);
        let index = q.gate.select(ctx, Constant(fe(step as i128)), zero, active);
        // CY[I] = CX[I]
        let value: Vec<Cell> = (0..PARTS).map(|part| {
            let component: Vec<Cell> = (0..NX).map(|i| cx[PARTS * i + part]).collect();
            q.index(ctx, &component, index)
        }).collect();
        let index_bound = q.bound(ctx, index);
        let position = q.wrap_index(ctx, index_bound, NY);
        for j in 0..NY {
            let hit = q.eq(ctx, position, Constant(fe(j as i128)));
            let write = q.gate.and(ctx, active, hit);
            for part in 0..PARTS {
                fast_cy[PARTS * j + part] = q.gate.select(ctx, value[part], fast_cy[PARTS * j + part], write);
            }
        }
    }
    // else: IX = 0; IY = 0
    let mut ix = zero;
    let mut iy = zero;
    // if INCX < 0: IX = (-N + 1) * INCX
    let xb = q.bound(ctx, incx);
    let negative_x = q.lt_const(ctx, xb, 0);
    let minus_n = q.gate.neg(ctx, n);
    let shift = q.add(ctx, minus_n, Constant(Fr::ONE));
    let start = q.gate.mul(ctx, shift, incx);
    ix = q.gate.select(ctx, start, ix, negative_x);
    // if INCY < 0: IY = (-N + 1) * INCY
    let yb = q.bound(ctx, incy);
    let negative_y = q.lt_const(ctx, yb, 0);
    let minus_n = q.gate.neg(ctx, n);
    let shift = q.add(ctx, minus_n, Constant(Fr::ONE));
    let start = q.gate.mul(ctx, shift, incy);
    iy = q.gate.select(ctx, start, iy, negative_y);
    let mut slow_cy = cy.clone();
    for step in 0..MAX_STEPS {
        let nb = q.bound(ctx, n);
        let running = q.gt_const(ctx, nb, step as i128);
        let active = q.gate.and(ctx, slow, running);
        let safe_x = q.gate.select(ctx, ix, zero, active);
        let safe_y = q.gate.select(ctx, iy, zero, active);
        // CY[IY] = CX[IX]
        let value: Vec<Cell> = (0..PARTS).map(|part| {
            let component: Vec<Cell> = (0..NX).map(|i| cx[PARTS * i + part]).collect();
            q.index(ctx, &component, safe_x)
        }).collect();
        let index_bound = q.bound(ctx, safe_y);
        let position = q.wrap_index(ctx, index_bound, NY);
        for j in 0..NY {
            let hit = q.eq(ctx, position, Constant(fe(j as i128)));
            let write = q.gate.and(ctx, active, hit);
            for part in 0..PARTS {
                slow_cy[PARTS * j + part] = q.gate.select(ctx, value[part], slow_cy[PARTS * j + part], write);
            }
        }
        // IX = IX + INCX; IY = IY + INCY
        let next = q.add(ctx, ix, incx);
        ix = q.gate.select(ctx, next, ix, active);
        let next = q.add(ctx, iy, incy);
        iy = q.gate.select(ctx, next, iy, active);
    }
    let result: Vec<Cell> = (0..PARTS * NY).map(|i| {
        let branch = q.gate.select(ctx, fast_cy[i], slow_cy[i], unit);
        q.gate.select(ctx, branch, cy[i], positive)
    }).collect();
    out.floats(&result);
    Ok(())
}
