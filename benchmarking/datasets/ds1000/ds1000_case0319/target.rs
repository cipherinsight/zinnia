//! DS-1000 case 0319, halo2-base baseline.
//! Kernel: `c = np.empty_like(p); c[p] = np.arange(8); a[c, :, :]` on
//! `a: Float[8, 6, 8]`, `permutation: Integer[8]`.
//! Precondition (enforced): `permutation` is a permutation of 0..8, the
//! kernel's `@requires(is_permutation(permutation))` (author, 2026-10-06: the
//! input is stated as a permutation on both sides). Any other input has no
//! proof (NumPy would raise, wrap a negative index, or leave entries of `c`
//! uninitialized).
//! Output plane j is plane c[j] of `a`, the k with permutation[k] = j: the
//! one-hot hit[j][k] = [p_k == j] (one
//! zero test, made once and shared by the 48 cells of the plane) selects it.
//! Each k hits at most one j, so "every plane j has exactly one hit" proves
//! the precondition.

use crate::prelude::*;

const N: usize = 8;
const PLANE: usize = 6 * 8;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, perm) = (inputs.floats("a", &[N, 6, 8])?, inputs.ints("permutation", &[N])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let perm = q.load_ints(ctx, &perm);
    let mut result = Vec::with_capacity(N * PLANE);
    for j in 0..N {
        let hit: Vec<Cell> = perm.iter().map(|p| q.eq(ctx, *p, Constant(fe(j as i128)))).collect();
        let total = q.sum(ctx, hit.iter().copied());
        q.gate.assert_is_const(ctx, &total, &Fr::ONE);
        for m in 0..PLANE {
            let column = (0..N).map(|k| Existing(a[k * PLANE + m]));
            result.push(q.gate.inner_product(ctx, hit.iter().map(|h| Existing(*h)), column));
        }
    }
    out.floats(&result);
    Ok(())
}
