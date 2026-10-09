// pythran_matmul__matrix_multiply, Noir literal baseline.
// Literal translation of kernel.py (LITERAL_BASELINES.md).
// Source:
//   new_matrix = zero(len(m0), len(m1[0]))
//   for i in range(len(m0)):
//       for j in range(len(m1[0])):
//           r = 0
//           for k in range(len(m1)):
//               r += m0[i][k] * m1[k][j]
//           new_matrix[i][j] = r
// for m0: Float[N, K], m1: Float[K, P].
// Translation choices:
// - zero(n, m) only fixes the output's shape (N rows by P columns at this
//   fixture, since N == P); it computes no values of its own.
// - This is an explicit Python loop accumulating products, not a single
//   np.dot/@ call, so L5 rescales each product separately before adding it
//   to r (no dot-product reassociation).
// @outputs: float[N * P]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = m0_0
global K: u32 = 16; // = m0_1
global P: u32 = 16; // = m1_1

fn main(m0: [[Field; K]; N], m1: [[Field; P]; K]) -> pub [[Field; P]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(m0[i]);
    }
    for k in 0..K {
        check_inputs(m1[k]);
    }
    let mut result: [[Field; P]; N] = [[0; P]; N];
    for i in 0..N {
        for j in 0..P {
            // r = 0; for k in range(K): r += m0[i][k] * m1[k][j]
            let mut r: Field = 0;
            for k in 0..K {
                r += mul(m0[i][k], m1[k][j]);
            }
            result[i][j] = r;
        }
    }
    result
}
