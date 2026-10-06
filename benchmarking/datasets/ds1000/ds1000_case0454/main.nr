// DS-1000 case 0454, Noir baseline.
// Kernel: `X / np.abs(X).max(axis=1).reshape(-1, 1)` on Integer[5, 4].
// The row norm max|x| is max_of over the 8 values x and -x of the row. `/` is
// NumPy's true division, so each output is the Q32 quotient
// floor(x * 2^32 / linf) of two integers (positive-divisor floor division).
// A zero row (linf = 0, NumPy gives nan) has no proof.
// @outputs: float[20]

use noir_hw::{divmod_pos, max_of, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut signed: [Field; 8] = [0; 8];
        for j in 0..4 {
            signed[2 * j] = X[i * 4 + j];
            signed[2 * j + 1] = -X[i * 4 + j];
        }
        let linf = max_of(signed);
        for j in 0..4 {
            result[i * 4 + j] = divmod_pos(X[i * 4 + j] * SCALE, linf).0;
        }
    }
    result
}
