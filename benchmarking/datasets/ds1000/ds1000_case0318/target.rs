//! DS-1000 case 0318, halo2-base baseline.
//! Kernel: `c = np.empty_like(p); c[p] = np.arange(6); a[:, c]` on
//! `a: Float[8, 6]`, `permutation: Integer[6]`.
//! Precondition (enforced): `permutation` is a permutation of 0..6 after
//! NumPy's negative-index wrap. Otherwise NumPy raises (an index outside
//! [-6, 6)) or leaves entries of `c` uninitialized (`np.empty`), and there is
//! no proof.
//! Output column j is column c[j] of `a`, the k with permutation[k] = j: the
//! one-hot hit[j][k] = [(p_k - j)(p_k - j + 6) == 0] (p_k is j or j - 6; one
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
    let perm_sq: Vec<Cell> = perm.iter().map(|p| q.gate.mul(ctx, *p, *p)).collect();
    let mut hit = Vec::with_capacity(N); // hit[j][k]
    for j in 0..N {
        let row: Vec<Cell> = perm.iter().zip(&perm_sq).map(|(p, sq)| wraps_to(&q, ctx, *p, *sq, j, N)).collect();
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

/// [idx == pos] + [idx == pos - len] (idx is a NumPy index of position `pos` on
/// an axis of length `len`) as one zero test: (idx - pos)(idx - pos + len) = 0,
/// i.e. idx^2 + (len - 2 pos) idx == pos (len - pos). `idx_sq` is idx^2.
fn wraps_to(q: &Q32, ctx: &mut Ctx, idx: Cell, idx_sq: Cell, pos: usize, len: usize) -> Cell {
    let (pos, len) = (pos as i128, len as i128);
    let t = q.gate.mul_add(ctx, idx, Constant(fe(len - 2 * pos)), idx_sq);
    q.eq(ctx, t, Constant(fe(pos * (len - pos))))
}
