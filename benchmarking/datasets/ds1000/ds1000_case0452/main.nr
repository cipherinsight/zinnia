// DS-1000 case 0452, Noir baseline.
// Kernel: `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on Integer[5, 4].
// `/` is NumPy's true division, so each output is the Q32 quotient
// floor(x * 2^32 / l1) of two integers (positive-divisor floor division).
// |x| is noir_hw::abs (cheaper here than max_of([x, -x])). A zero row (l1 = 0,
// NumPy gives nan) has no proof.
// @outputs: float[20]

use noir_hw::{abs, divmod_pos, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut l1: Field = 0;
        for j in 0..4 {
            l1 += abs(X[i * 4 + j]);
        }
        for j in 0..4 {
            result[i * 4 + j] = divmod_pos(X[i * 4 + j] * SCALE, l1).0;
        }
    }
    result
}
