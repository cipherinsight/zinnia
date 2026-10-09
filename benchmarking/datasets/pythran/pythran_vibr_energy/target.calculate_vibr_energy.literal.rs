// pythran_vibr_energy__calculate_vibr_energy: literal translation of kernel.py (LITERAL_BASELINES.md).
// Full elementwise intermediates, negation before multiplication. Exp uses the same canonical shared gadget as optimized.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = harmonic_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let harmonic = inputs.floats("harmonic", &[N])?;
    let anharmonic = inputs.floats("anharmonic", &[N])?;
    let i = inputs.floats("i", &[N])?;
    let q = Q32::new(builder); let ctx = builder.main(0);
    let harmonic = q.load_floats(ctx, &harmonic); let anharmonic = q.load_floats(ctx, &anharmonic); let i = q.load_floats(ctx, &i);
    let negative: Vec<Cell> = harmonic.iter().map(|v| q.neg(ctx, *v)).collect();
    let linear: Vec<Cell> = (0..N).map(|k| q.mul(ctx, negative[k], i[k]).cell).collect();
    let square: Vec<Cell> = i.iter().map(|v| q.pow_const(ctx, *v, 2)).collect();
    let quadratic: Vec<Cell> = (0..N).map(|k| q.mul(ctx, anharmonic[k], square[k]).cell).collect();
    let exponent: Vec<Cell> = (0..N).map(|k| q.sub(ctx, linear[k], quadratic[k])).collect();
    let energies: Vec<Cell> = exponent.iter().map(|v| { let x = q.bound(ctx, *v); q.canonical_exp(ctx, x).cell }).collect();
    out.floats(&energies); Ok(())
}
