// DS-1000 case 0453, Noir baseline.
// Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on Integer[5, 4].
// The squares and their row sum stay integers; the norm is the Q32 square root
// of that sum, and `/` is NumPy's true division (a Q32 float division by the
// non-negative norm). No proof for a zero row (NumPy gives nan) or a row whose
// sum of squares reaches 2^62 (its norm leaves the Q32 window).
// Every input with a proof therefore has |x| < 2^31, and the circuit checks
// exactly that (x + 2^31 in [0, 2^32)): it proves the inputs are integers, so
// the squares and sums are exact. Without it, field elements x = X / 2^64
// whose squares wrap the field give "normalized" values near 2^31.
// @outputs: float[20]

use noir_hw::{divmod_pos, from_int, sqrt, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut sq_sum: Field = 0;
        for j in 0..4 {
            (X[i * 4 + j] + 2147483648).assert_max_bit_size::<32>();
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
