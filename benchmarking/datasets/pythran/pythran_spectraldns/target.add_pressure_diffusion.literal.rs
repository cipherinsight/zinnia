//! pythran_spectraldns__add_pressure_diffusion: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize full complex arrays per expression, with no CSE or dead-write elimination.
//! Dead-only inputs use shared integer limbs. Discarded products retain Q32 floors
//! through small shared divisions and exact field reconstruction, without output
//! windows absent from optimized. Low limbs carry rounding through the second product.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.add_pressure_diffusion.json.
const A1: usize = 2; // = dU_1
const A2: usize = 5; // = dU_2
const A3: usize = 7; // = dU_3
const POINTS: usize = A1 * A2 * A3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let du = inputs.complexes("dU", &[3, A1, A2, A3])?;
    let uh = inputs.complexes("U_hat", &[3, A1, A2, A3])?;
    let k2 = inputs.floats("K2", &[A1, A2, A3])?;
    let k = inputs.floats("K", &[3, A1, A2, A3])?;
    let ko = inputs.floats("K_over_K2", &[3, A1, A2, A3])?;
    let nu = inputs.float("nu")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let mut du = q.load_floats(ctx, &du);
    let uh: Vec<_> = uh.iter().map(|v| load_discarded(&q, ctx, *v)).collect();
    let k2: Vec<_> = k2.iter().map(|v| load_discarded(&q, ctx, *v)).collect();
    let k: Vec<_> = k.iter().map(|v| load_discarded(&q, ctx, *v)).collect();
    let ko = q.load_floats(ctx, &ko);
    let nu = load_discarded(&q, ctx, nu);
    // du0*k_0 + du1*k_1 + du2*k_2, left-associated, whole arrays.
    let first: Vec<Cell> = (0..2 * POINTS)
        .map(|i| q.mul(ctx, du[i], ko[i / 2]).cell)
        .collect();
    let second: Vec<Cell> = (0..2 * POINTS)
        .map(|i| q.mul(ctx, du[2 * POINTS + i], ko[POINTS + i / 2]).cell)
        .collect();
    let sum: Vec<Cell> = (0..2 * POINTS)
        .map(|i| q.add(ctx, first[i], second[i]))
        .collect();
    let third: Vec<Cell> = (0..2 * POINTS)
        .map(|i| q.mul(ctx, du[4 * POINTS + i], ko[2 * POINTS + i / 2]).cell)
        .collect();
    let ph: Vec<Cell> = (0..2 * POINTS)
        .map(|i| q.add(ctx, sum[i], third[i]))
        .collect();
    // dU[:] = dU - P_hat*K - nu*K2*U_hat (computed although not returned).
    let pressure: Vec<Cell> = (0..6 * POINTS)
        .map(|i| {
            let value = ph[i % (2 * POINTS)];
            let low = q.floor_div_const(ctx, value, SCALE as u64).1;
            discarded_product(&q, ctx, value, k[i / 2].0, low, k[i / 2].1).0
        })
        .collect();
    let difference: Vec<Cell> = (0..6 * POINTS)
        .map(|i| q.sub(ctx, du[i], pressure[i]))
        .collect();
    let viscosity: Vec<_> = k2.iter().map(|v| {
        let (value, low_quotient) = discarded_product(&q, ctx, nu.0, v.0, nu.1, v.1);
        let cross0 = q.gate.mul(ctx, nu.1, v.2);
        let cross1 = q.gate.mul(ctx, nu.2, v.1);
        let sum = q.add(ctx, low_quotient, cross0);
        let sum = q.add(ctx, sum, cross1);
        let low = q.floor_div_const(ctx, sum, SCALE as u64).1;
        (value, low)
    }).collect();
    let diffusion: Vec<Cell> = (0..6 * POINTS)
        .map(|i| {
            let v = viscosity[(i / 2) % POINTS];
            discarded_product(&q, ctx, v.0, uh[i].0, v.1, uh[i].1).0
        })
        .collect();
    for i in 0..6 * POINTS {
        du[i] = q.sub(ctx, difference[i], diffusion[i]);
    }
    out.floats(&ph);
    Ok(())
}

// Input encoding only: four base-2^32 limbs reconstruct every quantized i128.
// The high limb is signed; low limbs also supply the rounding residues.
fn load_discarded(q: &Q32, ctx: &mut Ctx, value: f64) -> (Cell, Cell, Cell) {
    let code = quantize(value);
    let mask = SCALE - 1;
    let low = q.load_int(ctx, (code & mask) as i64);
    let next = q.load_int(ctx, ((code >> 32) & mask) as i64);
    let upper = q.load_int(ctx, ((code >> 64) & mask) as i64);
    let high = q.load_int(ctx, (code >> 96) as i64);
    let mut whole = high;
    for limb in [upper, next, low] {
        whole = q.gate.mul_add(ctx, whole, Constant(fe(SCALE)), limb);
    }
    (whole, low, next)
}

// For actual integers a,b and their low limbs, r=(a*b) mod SCALE is small.
// (a*b-r)/SCALE is then exactly floor(a*b/SCALE), represented in the field.
// Keep the small-product quotient for a following product's low-limb calculation.
// Shared floor division supplies every quotient witness; there is no local hint.
fn discarded_product(q: &Q32, ctx: &mut Ctx, a: Cell, b: Cell, a_low: Cell, b_low: Cell) -> (Cell, Cell) {
    let raw = q.gate.mul(ctx, a, b);
    let low_raw = q.gate.mul(ctx, a_low, b_low);
    let (low_quotient, remainder) = q.floor_div_const(ctx, low_raw, SCALE as u64);
    let divisible = q.sub(ctx, raw, remainder);
    let result = q.gate.mul(ctx, divisible, Constant(fe(SCALE).invert().unwrap()));
    (result, low_quotient.cell)
}
