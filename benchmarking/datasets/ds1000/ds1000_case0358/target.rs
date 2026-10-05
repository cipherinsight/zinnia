//! DS-1000 case 0358, halo2-base baseline.
//! Kernel: `b[np.argsort(a.sum(axis=(1, 2))), :, :]` on `a: Float[3, 3, 3]`, `b: Integer[3, 3, 3]`.
//! The three matrices of `b` in the ascending order of the sums of the matrices
//! of `a`. NumPy sorts three elements by insertion, which is stable: ties keep
//! their order. From the three comparisons `s01 = s1 < s0`, `s02 = s2 < s0`,
//! `s12 = s2 < s1`, the one-hot selectors of the first matrix are
//! `(1 - f1 - f2, s01 (1 - s12), s02 s12)` and of the last
//! `(s01 s02, (1 - s01) s12, 1 - l0 - l1)`; the middle selector is what remains.
//! Each output cell is a selector's inner product with the three `b` cells.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b) = (inputs.floats("a", &[3, 3, 3])?, inputs.ints("b", &[3, 3, 3])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let b = q.load_ints(ctx, &b);
    let s: Vec<I64> = a
        .chunks(9)
        .map(|m| {
            let total = q.sum(ctx, m.iter().copied());
            q.bound(ctx, total)
        })
        .collect();
    let s01 = q.lt(ctx, s[1], s[0]);
    let s02 = q.lt(ctx, s[2], s[0]);
    let s12 = q.lt(ctx, s[2], s[1]);

    let s01_s12 = q.gate.and(ctx, s01, s12);
    let f1 = q.sub(ctx, s01, s01_s12);
    let f2 = q.gate.and(ctx, s02, s12);
    let f0 = q.sub(ctx, Constant(Fr::ONE), f1);
    let f0 = q.sub(ctx, f0, f2);
    let l0 = q.gate.and(ctx, s01, s02);
    let l1 = q.sub(ctx, s12, s01_s12);
    let l2 = q.sub(ctx, Constant(Fr::ONE), l0);
    let l2 = q.sub(ctx, l2, l1);
    let first = [f0, f1, f2];
    let last = [l0, l1, l2];
    let middle: Vec<Cell> = (0..3)
        .map(|k| {
            let m = q.sub(ctx, Constant(Fr::ONE), first[k]);
            q.sub(ctx, m, last[k])
        })
        .collect();

    let mut result = Vec::with_capacity(27);
    for selector in [&first[..], &middle[..], &last[..]] {
        for p in 0..9 {
            let cells = [b[p], b[9 + p], b[18 + p]].map(Existing);
            result.push(q.gate.inner_product(ctx, selector.iter().copied(), cells));
        }
    }
    out.ints(&result);
    Ok(())
}
