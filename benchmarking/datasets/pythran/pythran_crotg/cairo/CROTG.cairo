//! Complex Givens rotation, preserving the reference's magnitude rounding.
// unused parameters: C, S
// output: (c32,c32,q32,c32)
use crate::io::{ArgsTrait, Word};
use crate::q32::{C32, C32Trait, Q32Trait, ZERO, C_ONE};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let ca: C32 = args.scalar();
    let cb: C32 = args.scalar();
    let abs_ca = ca.abs();
    let (result, c, s) = if abs_ca == ZERO {
        (cb, ZERO, C_ONE)
    } else {
        let scale = abs_ca + cb.abs();
        let norm = scale * (ca.div_real(scale).abs().powi(2) + cb.div_real(scale).abs().powi(2)).sqrt();
        let alpha = ca.div_real(abs_ca);
        (alpha.scale(norm), abs_ca / norm, (alpha * cb.conj()).div_real(norm))
    };
    let mut out = args.output();
    Word::write(result, ref out);
    Word::write(cb, ref out);
    Word::write(c, ref out);
    Word::write(s, ref out);
    out
}
