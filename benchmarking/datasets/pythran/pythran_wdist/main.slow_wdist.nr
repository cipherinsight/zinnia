// pythran_wdist__slow_wdist (Pythran wdist), Noir baseline.
// Kernel: D[ii, jj] = sqrt(sum_k ((A[k, ii] - B[k, jj]) / W[k, ii]) ** 2) on
// A: Float[K, MM], B: Float[K, NN], W: Float[K, MM]. Each quotient is a float
// division (NumPy materializes wdiff); (wdiff ** 2).sum() is one reduction,
// rescaled once (NPB_BASELINES.md section 5, rule 3); the root is the exact
// floor square root. A divisor W[k, ii] serves every jj, so its sign test runs
// once (`abs`, then `div_by_abs`: `div` with the magnitude hoisted).
// @outputs: float[MM * NN]

use noir_hw::{abs, check_inputs, div_by_abs, rescale, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global K: u32 = 2; // = A_0
global MM: u32 = 3; // = A_1
global NN: u32 = 3; // = B_1

fn main(A: [[Field; MM]; K], B: [[Field; NN]; K], W: [[Field; MM]; K]) -> pub [[Field; NN]; MM] {
    // Input domain: every input element is a Q32 code in the window.
    for k in 0..K {
        check_inputs(A[k]);
        check_inputs(B[k]);
        check_inputs(W[k]);
    }
    let mut w_abs: [[Field; MM]; K] = [[0; MM]; K];
    for k in 0..K {
        for ii in 0..MM {
            w_abs[k][ii] = abs(W[k][ii]);
        }
    }
    let mut d: [[Field; NN]; MM] = [[0; NN]; MM];
    for ii in 0..MM {
        for jj in 0..NN {
            let mut acc: Field = 0;
            for k in 0..K {
                let wdiff = div_by_abs(A[k][ii] - B[k][jj], W[k][ii], w_abs[k][ii]);
                acc += wdiff * wdiff;
            }
            d[ii][jj] = sqrt(rescale(acc));
        }
    }
    d
}
