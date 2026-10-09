//! pythran_spectraldns__add_pressure_diffusion (Pythran spectraldns), halo2-base baseline.
//! Kernel: `P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2` with
//! `du_k = dU[k]` (dU: Complex[3, A1, A2, A3]) and `k_k = K_over_K2[k]`
//! (Float[3, A1, A2, A3]), then a write of `dU` that reaches no output; returns
//! `P_hat` (Complex[A1, A2, A3]).
//! The leading 3 is the vector field's, not a size. Only `dU` and `K_over_K2`
//! reach the output, so only they are loaded (`P_hat` is overwritten first;
//! `U_hat`, `K2`, `K`, `nu` feed the dead write).
//! A complex times a real value is two rounded products (the real and
//! imaginary parts of the complex product with a zero imaginary part, each one
//! rescale); NumPy materializes the three product arrays before adding them,
//! so each is rounded.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.add_pressure_diffusion.json.
const A1: usize = 2; // = dU_1
const A2: usize = 5; // = dU_2
const A3: usize = 7; // = dU_3
const POINTS: usize = A1 * A2 * A3;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let du = inputs.complexes("dU", &[3, A1, A2, A3])?;
    let k_over_k2 = inputs.floats("K_over_K2", &[3, A1, A2, A3])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let du = q.load_floats(ctx, &du); // interleaved real and imaginary parts
    let k_over_k2 = q.load_floats(ctx, &k_over_k2);
    let mut p_hat = Vec::with_capacity(2 * POINTS);
    for p in 0..POINTS {
        for part in 0..2 {
            let terms: Vec<Cell> =
                (0..3).map(|k| q.mul(ctx, du[2 * (k * POINTS + p) + part], k_over_k2[k * POINTS + p]).cell).collect();
            p_hat.push(q.sum(ctx, terms));
        }
    }
    out.floats(&p_hat);
    Ok(())
}
