//! DS-1000 case 0491, halo2-base baseline.
//! Kernel: `(~np.isclose(s1, s2)).sum()` on two `Float[6, 1]`.
//! `isclose` is NumPy's `|a - b| <= atol + rtol * |b|` with the defaults
//! `rtol = 1e-5`, `atol = 1e-8`. Both magnitudes are `q.abs_int64`: `|s1 - s2|`
//! or `|s2|` equal to `2^63` in Q32 code (`|i64::MIN|`, which overflows int64)
//! has no proof, as in Zinnia.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (s1, s2) = (inputs.floats("s1", &[6, 1])?, inputs.floats("s2", &[6, 1])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (s1, s2) = (q.load_floats(ctx, &s1), q.load_floats(ctx, &s2));
    let mut not_close = Vec::with_capacity(6);
    for (&a, &b) in s1.iter().zip(&s2) {
        let diff = q.sub(ctx, a, b);
        let abs_diff = q.abs_int64(ctx, diff);
        let abs_b = q.abs_int64(ctx, b);
        let tol = q.mul(ctx, abs_b, Constant(fe(quantize(1e-5))));
        let tol = q.add(ctx, tol.cell, Constant(fe(quantize(1e-8))));
        let slack = q.sub(ctx, tol, abs_diff); // close <=> slack >= 0
        let slack = q.bound(ctx, slack);
        not_close.push(q.is_neg(ctx, slack));
    }
    let count = q.sum(ctx, not_close);
    out.int(count);
    Ok(())
}
