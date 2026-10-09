//! pythran_spectraldns__add_pressure_diffusion: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Materialize full complex arrays per expression, with no CSE or dead-write elimination.
//! Real-complex multiplication is two Q32 products; multiplication by 1j uses the complex definition.

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
    let uh = q.load_floats(ctx, &uh);
    let k2 = q.load_floats(ctx, &k2);
    let k = q.load_floats(ctx, &k);
    let ko = q.load_floats(ctx, &ko);
    let nu = q.load_float(ctx, nu);
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
        .map(|i| q.mul(ctx, ph[i % (2 * POINTS)], k[i / 2]).cell)
        .collect();
    let difference: Vec<Cell> = (0..6 * POINTS)
        .map(|i| q.sub(ctx, du[i], pressure[i]))
        .collect();
    let viscosity: Vec<Cell> = k2.iter().map(|v| q.mul(ctx, nu, *v).cell).collect();
    let diffusion: Vec<Cell> = (0..6 * POINTS)
        .map(|i| q.mul(ctx, viscosity[(i / 2) % POINTS], uh[i]).cell)
        .collect();
    for i in 0..6 * POINTS {
        du[i] = q.sub(ctx, difference[i], diffusion[i]);
    }
    out.floats(&ph);
    Ok(())
}
