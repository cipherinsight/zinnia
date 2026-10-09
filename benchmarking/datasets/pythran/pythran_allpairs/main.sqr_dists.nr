// pythran_allpairs__sqr_dists (Pythran allpairs), Noir baseline.
// Kernel: np.array([[np.sum((x - y) ** 2) for x in X] for y in Y]) on
// X: Float[M, N], Y: Float[P, N]; the output is P x M, row j holding the
// squared distances from Y[j] to every row of X.
// Preserve pairwise differences and round each square before summing.
// @outputs: float[P * M]

use noir_hw::{check_inputs, mul};

// Sizes (hw_check.sizes): from fixture.sqr_dists.json.
global M: u32 = 16; // = X_0
global N: u32 = 16; // = X_1
global P: u32 = 16; // = Y_0

fn main(X: [[Field; N]; M], Y: [[Field; N]; P]) -> pub [[Field; M]; P] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(X[i]);
    }
    for j in 0..P {
        check_inputs(Y[j]);
    }
    let mut result: [[Field; M]; P] = [[0; M]; P];
    for j in 0..P {
        for i in 0..M {
            let mut acc: Field = 0;
            for k in 0..N {
                let diff = X[i][k] - Y[j][k];
                acc += mul(diff, diff);
            }
            result[j][i] = acc;
        }
    }
    result
}
