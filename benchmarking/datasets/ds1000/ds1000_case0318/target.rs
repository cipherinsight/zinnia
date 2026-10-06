//! DS-1000 case 0318, halo2-base baseline.
//! Kernel: `c = np.empty_like(p); c[p] = np.arange(6); a[:, c]` on
//! `a: Float[8, 6]`, `permutation: Integer[6]`.
//! Precondition (enforced): `permutation` is a permutation of 0..6, the
//! kernel's `@requires(is_permutation(permutation))` (author, 2026-10-06: the
//! input is stated as a permutation on both sides). Any other input has no
//! proof (NumPy would raise, wrap a negative index, or leave entries of `c`
//! uninitialized).
//! Output column j is column c[j] of `a`, the k with permutation[k] = j: the
//! one-hot hit[j][k] = [p_k == j] (one
//! zero test, made once and shared by the 8 rows) selects it. Each k hits at
//! most one j, so "every column j has exactly one hit" proves the precondition.

use crate::prelude::*;

const N: usize = 6;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, perm) = (inputs.floats("a", &[8, N])?, inputs.ints("permutation", &[N])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let perm = q.load_ints(ctx, &perm);
    let mut hit = Vec::with_capacity(N); // hit[j][k]
    for j in 0..N {
        let row: Vec<Cell> = perm.iter().map(|p| q.eq(ctx, *p, Constant(fe(j as i128)))).collect();
        let total = q.sum(ctx, row.iter().copied());
        q.gate.assert_is_const(ctx, &total, &Fr::ONE);
        hit.push(row);
    }
    let mut result = Vec::with_capacity(8 * N);
    for a_row in a.chunks(N) {
        for hit_j in &hit {
            let sel = hit_j.iter().map(|h| Existing(*h));
            result.push(q.gate.inner_product(ctx, sel, a_row.iter().map(|x| Existing(*x))));
        }
    }
    out.floats(&result);
    Ok(())
}
