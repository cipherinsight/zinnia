// pythran_pairwise__pairwise: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute all ordered pairs including diagonal; omit constant diagonal only from public outputs.

// @outputs: const float[1] = [0]; repeat[r < M - 1](float[M] | const float[1] = [0])

use noir_hw::{check_inputs, mul, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global M: u32 = 16; // = X_0
global N: u32 = 8; // = X_1
global OFF_DIAGONAL: u32 = M * (M - 1);

fn main(X: [[Field; N]; M]) -> pub [Field; OFF_DIAGONAL] {
    for i in 0..M { check_inputs(X[i]); }
    let mut distances = [[0; M]; M];
    for i in 0..M { for j in 0..M {
        let mut d: Field = 0;
        for k in 0..N {
            let tmp = X[i][k] - X[j][k];
            d += mul(tmp, tmp);
        }
        d.assert_max_bit_size::<64>();
        distances[i][j] = sqrt(d);
    } }
    let mut result = [0; OFF_DIAGONAL];
    for i in 0..M { for j in 0..M { if i != j {
        let pos = if j<i { i*(M-1)+j } else { i*(M-1)+j-1 };
        result[pos] = distances[i][j];
    } } }
    result
}
