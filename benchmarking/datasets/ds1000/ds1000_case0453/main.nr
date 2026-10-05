// DS-1000 case 0453, Noir baseline.
// Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on Integer[5, 4].
// The squares and their row sum stay integers; the norm is the Q32 square root
// of that sum, and `/` is NumPy's true division (a Q32 float division by the
// non-negative norm). No proof for a zero row (NumPy gives nan) or a row whose
// sum of squares reaches 2^62 (its norm leaves the Q32 window).
// @outputs: float[20]

use noir_hw::{divmod_pos, from_int, sqrt, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut sq_sum: Field = 0;
        for j in 0..4 {
            sq_sum += X[i * 4 + j] * X[i * 4 + j];
        }
        let l2 = sqrt(from_int(sq_sum));
        for j in 0..4 {
            // floor(float(x) * 2^32 / l2)
            result[i * 4 + j] = divmod_pos(from_int(X[i * 4 + j]) * SCALE, l2).0;
        }
    }
    result
}
