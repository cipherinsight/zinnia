// pythran_pairwise__pairwise (Pythran pairwise), Noir baseline.
// Kernel: D[i, j] = sqrt(sum_k (X[i, k] - X[j, k])^2) on X: Float[M, N].
// Each sum of squares is one reduction, rescaled once (NPB_BASELINES.md
// section 5, rule 3), and its root the exact floor square root. D[i, j] and
// D[j, i] are the same computation, so each pair is computed once and output
// twice; the diagonal is sqrt(0) = 0 for every input, a constant output (not
// public, CONVENTIONS 1.3). The public outputs are the off-diagonal elements
// in row-major order.
// @outputs: const float[1] = [0]; repeat[r < M - 1](float[M] | const float[1] = [0])

use noir_hw::{check_inputs, rescale, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 16; // = X_0
global N: u32 = 8; // = X_1
global OFF_DIAGONAL: u32 = M * (M - 1);

fn main(X: [[Field; N]; M]) -> pub [Field; OFF_DIAGONAL] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..M {
        check_inputs(X[i]);
    }
    let mut d: [Field; OFF_DIAGONAL] = [0; OFF_DIAGONAL];
    for i in 0..M {
        for j in i + 1..M {
            let mut acc: Field = 0;
            for k in 0..N {
                let diff = X[i][k] - X[j][k];
                acc += diff * diff;
            }
            let root = sqrt(rescale(acc));
            // Row-major position among the off-diagonal elements: (i, j)
            // follows i + 1 diagonal elements, (j, i) follows j.
            d[i * M + j - i - 1] = root;
            d[j * M + i - j] = root;
        }
    }
    d
}
