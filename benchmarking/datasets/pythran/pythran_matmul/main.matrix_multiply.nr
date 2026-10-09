// pythran_matmul__matrix_multiply (Pythran matmul), Noir baseline.
// Kernel: the triple loop r += m0[i][k] * m1[k][j] for m0: Float[N, K],
// m1: Float[K, P], returning the N x P products as nested lists.
// Each output accumulates its products into one variable, so it is one
// reduction rescaled once (NPB_BASELINES.md section 5.3).
// @outputs: float[N * P]

use noir_hw::{check_inputs, rescale};

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
            let mut acc: Field = 0;
            for k in 0..K {
                acc += m0[i][k] * m1[k][j];
            }
            result[i][j] = rescale(acc);
        }
    }
    result
}
