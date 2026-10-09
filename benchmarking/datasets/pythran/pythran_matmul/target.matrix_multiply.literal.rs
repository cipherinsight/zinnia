//! pythran_matmul__matrix_multiply, halo2-base literal baseline.
//! Literal translation of kernel.py (LITERAL_BASELINES.md).
//! Source:
//!   new_matrix = zero(len(m0), len(m1[0]))
//!   for i in range(len(m0)):
//!       for j in range(len(m1[0])):
//!           r = 0
//!           for k in range(len(m1)):
//!               r += m0[i][k] * m1[k][j]
//!           new_matrix[i][j] = r
//! for m0: Float[N, K], m1: Float[K, P].
//! Translation choices:
//! - `zero(n, m)` only fixes the output's shape (a compile-time constant,
//!   `N` rows by `P` columns at this fixture, since `N == P`); it computes no
//!   values of its own.
//! - This is an explicit Python loop accumulating products, not a single
//!   `np.dot`/`@` call, so L5 rescales each product separately before adding
//!   it to `r` (no dot-product reassociation, no fused one-rescale sum).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = m0_0
const K: usize = 16; // = m0_1
const P: usize = 16; // = m1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m0 = inputs.floats("m0", &[N, K])?;
    let m1 = inputs.floats("m1", &[K, P])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let m0 = q.load_floats(ctx, &m0);
    let m1 = q.load_floats(ctx, &m1);
    let mut result: Vec<Cell> = Vec::with_capacity(N * P);
    for i in 0..N {
        for j in 0..P {
            // r = 0; for k in range(K): r += m0[i][k] * m1[k][j]
            let products: Vec<Cell> = (0..K)
                .map(|k| q.mul(ctx, m0[i * K + k], m1[k * P + j]).cell)
                .collect();
            let r = q.sum(ctx, products);
            result.push(r);
        }
    }
    out.floats(&result);
    Ok(())
}
