//! DS-1000 case 0357, halo2-base baseline.
//! Kernel: `c = b[np.argsort(a, axis=0)[::-1], j, k]` on `a: Float[3, 3, 3]`, `b: Integer[3, 3, 3]`.
//! Column `c[:, j, k]` is `b[:, j, k]` in the reverse of the ascending order of
//! `a[:, j, k]`. NumPy sorts three elements by insertion, which is stable: ties
//! keep their order before the reversal. From the three comparisons
//! `s01 = a1 < a0`, `s02 = a2 < a0`, `s12 = a2 < a1`, the first element of the
//! ascending order is `s12 ? (s02 ? 2 : 0) : (s01 ? 1 : 0)` and the last
//! `s12 ? (s01 ? 0 : 1) : (s02 ? 0 : 2)`; their `b` values are selected directly
//! (no index gather), and the middle one is the column sum minus those two.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("a", &[3, 3, 3])?, inputs.ints("b", &[3, 3, 3])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let a = q.bound_all(ctx, &a);
    let b = q.load_ints(ctx, &b);
    let mut c = b.clone();
    for col in 0..9 {
        let (a0, a1, a2) = (a[col], a[9 + col], a[18 + col]);
        let (b0, b1, b2) = (b[col], b[9 + col], b[18 + col]);
        let s01 = q.lt(ctx, a1, a0);
        let s02 = q.lt(ctx, a2, a0);
        let s12 = q.lt(ctx, a2, a1);
        let first_if = q.gate.select(ctx, b2, b0, s02);
        let first_else = q.gate.select(ctx, b1, b0, s01);
        let first = q.gate.select(ctx, first_if, first_else, s12);
        let last_if = q.gate.select(ctx, b0, b1, s01);
        let last_else = q.gate.select(ctx, b0, b2, s02);
        let last = q.gate.select(ctx, last_if, last_else, s12);
        let total = q.sum(ctx, [b0, b1, b2]);
        let middle = q.sub(ctx, total, first);
        let middle = q.sub(ctx, middle, last);
        (c[col], c[9 + col], c[18 + col]) = (last, middle, first);
    }
    out.ints(&c);
    Ok(())
}
