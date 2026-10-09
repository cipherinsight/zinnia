//! pythran_spectraldns__cross2: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize full complex arrays per expression, with no CSE or dead-write elimination.
//! Real-complex multiplication is two Q32 products; multiplication by 1j uses the complex definition.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.cross2.json.
const A1: usize = 2; // = a_1
const A2: usize = 5; // = a_2
const A3: usize = 7; // = a_3
const POINTS: usize = A1 * A2 * A3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[3, A1, A2, A3])?;
    let b = inputs.complexes("b", &[3, A1, A2, A3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let b = q.load_floats(ctx, &b);
    let c = cross1(&q, ctx, &a, &b);
    let mut result = Vec::new();
    for i in 0..3 * POINTS {
        let re0 = q.mul(ctx, c[2 * i], Constant(fe(0))).cell;
        let im1 = q.mul(ctx, c[2 * i + 1], Constant(fe(SCALE))).cell;
        let re = q.sub(ctx, re0, im1);
        let re1 = q.mul(ctx, c[2 * i], Constant(fe(SCALE))).cell;
        let im0 = q.mul(ctx, c[2 * i + 1], Constant(fe(0))).cell;
        let im = q.add(ctx, re1, im0);
        result.extend([re, im]);
    }
    out.floats(&result);
    Ok(())
}
fn cross1(q: &Q32, ctx: &mut Ctx, a: &[Cell], b: &[Cell]) -> Vec<Cell> {
    let mut c = Vec::new();
    // Three source assignments, in order. The repeated a[0]*b[2] is computed twice.
    for (ak, bk, al, bl) in [(0, 2, 2, 1), (2, 0, 0, 2), (0, 1, 1, 0)] {
        let plus: Vec<Cell> = (0..2 * POINTS)
            .map(|i| {
                q.mul(ctx, a[ak * POINTS + i / 2], b[bk * 2 * POINTS + i])
                    .cell
            })
            .collect();
        let minus: Vec<Cell> = (0..2 * POINTS)
            .map(|i| {
                q.mul(ctx, a[al * POINTS + i / 2], b[bl * 2 * POINTS + i])
                    .cell
            })
            .collect();
        for i in 0..2 * POINTS {
            c.push(q.sub(ctx, plus[i], minus[i]));
        }
    }
    c
}
