//! pythran_vibr_energy__calculate_vibr_energy (Pythran vibr_energy), halo2-base baseline.
//! Strict comparison: transcendental calls use the frozen compiler arithmetic,
//! including its coefficients and each Q32 rounding point (canonical APIs).
//! Kernel: `numpy.exp(-harmonic * i - anharmonic * i ** 2)` on harmonic,
//! anharmonic, i: Float[N]. In NumPy's order: `(-harmonic) * i` (the negation
//! is exact, so the product rounds `-(harmonic * i)` down), `i ** 2` and
//! `anharmonic * i ** 2` are rounded products, the difference is exact, and
//! `q.canonical_exp` (the shared gadget, NPB_BASELINES.md section 5, rule 5) takes the
//! window-checked difference.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = harmonic_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let harmonic = inputs.floats("harmonic", &[N])?;
    let anharmonic = inputs.floats("anharmonic", &[N])?;
    let i = inputs.floats("i", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let harmonic = q.load_floats(ctx, &harmonic);
    let anharmonic = q.load_floats(ctx, &anharmonic);
    let i = q.load_floats(ctx, &i);
    let energies: Vec<Cell> = (0..N)
        .map(|k| {
            let minus_harmonic = q.neg(ctx, harmonic[k]);
            let linear = q.mul(ctx, minus_harmonic, i[k]);
            let square = q.mul(ctx, i[k], i[k]);
            let quadratic = q.mul(ctx, anharmonic[k], square.cell);
            let exponent = q.sub(ctx, linear.cell, quadratic.cell);
            let exponent = q.bound(ctx, exponent);
            q.canonical_exp(ctx, exponent).cell
        })
        .collect();
    out.floats(&energies);
    Ok(())
}
