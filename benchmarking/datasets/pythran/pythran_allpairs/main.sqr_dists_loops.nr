// pythran_allpairs__sqr_dists_loops (Pythran allpairs), Noir baseline.
// Kernel: D[i, j] = np.sum((X[i] - Y[j]) ** 2) for every row i of
// X: Float[M, N] and row j of Y: Float[P, N]; the output is D: M x P.
// Preserve pairwise differences and round each square before summing.
// @outputs: float[M * P]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.sqr_dists_loops.json.
global M: u32 = 16; // = X_0
global N: u32 = 16; // = X_1
global P: u32 = 16; // = Y_0

fn main(X: [[Field; N]; M], Y: [[Field; N]; P]) -> pub [[Field; P]; M] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(X[i]);
    }
    for j in 0..P {
        check_inputs(Y[j]);
    }
    let mut result: [[Field; P]; M] = [[0; P]; M];
    for i in 0..M {
        for j in 0..P {
            let mut acc: Field = 0;
            for k in 0..N {
                let diff = X[i][k] - Y[j][k];
                acc += mul(diff, diff);
            }
            result[i][j] = acc;
        }
    }
    result
}
