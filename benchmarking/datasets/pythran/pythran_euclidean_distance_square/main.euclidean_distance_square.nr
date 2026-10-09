// pythran_euclidean_distance_square__euclidean_distance_square (Pythran
// euclidean_distance_square), Noir baseline.
// Kernel: `-2 * np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, None]
// + np.sum(np.square(x2), axis=1)` on x1: Float[R1, D], x2: Float[R2, D].
// Each dot product and each row sum of squares is one reduction, rescaled
// once (`dot`; NPB_BASELINES.md section 5.3); `-2 *` is exact.
// @outputs: float[R1 * R2]

use noir_hw::{check_inputs, dot};

// Sizes (hw_check.sizes): from fixture.json.
global R1: u32 = 1; // = x1_0
global R2: u32 = 5; // = x2_0
global D: u32 = 3; // = x1_1

fn main(x1: [[Field; D]; R1], x2: [[Field; D]; R2]) -> pub [[Field; R2]; R1] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..R1 {
        check_inputs(x1[i]);
    }
    for j in 0..R2 {
        check_inputs(x2[j]);
    }
    let mut squares2: [Field; R2] = [0; R2];
    for j in 0..R2 {
        squares2[j] = dot(x2[j], x2[j]);
    }
    let mut result: [[Field; R2]; R1] = [[0; R2]; R1];
    for i in 0..R1 {
        let square1 = dot(x1[i], x1[i]);
        for j in 0..R2 {
            result[i][j] = -2 * dot(x1[i], x2[j]) + square1 + squares2[j];
        }
    }
    result
}
