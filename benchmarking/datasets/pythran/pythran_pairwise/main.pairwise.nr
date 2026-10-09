// pythran_pairwise__pairwise (Pythran pairwise), Noir baseline.
// Kernel: D[i, j] = sqrt(sum_k (X[i, k] - X[j, k])^2) on X: Float[M, N].
// Match every ordered pair, rounding each square before accumulation.
// Only the identically-zero diagonal folds; public cells are off-diagonal
// values in row-major order.
// @outputs: const float[1] = [0]; repeat[r < M - 1](float[M] | const float[1] = [0])

use noir_hw::{check_inputs, mul, sqrt};

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
        for j in 0..M {
            if i != j {
                let mut acc: Field = 0;
                for k in 0..N {
                    let diff = X[i][k] - X[j][k];
                    acc += mul(diff, diff);
                }
                // Both indices are loop constants; skip preceding diagonal.
                let pos = if j < i { i * (M - 1) + j } else { i * (M - 1) + j - 1 };
                d[pos] = sqrt(acc);
            }
        }
    }
    d
}
