// DS-1000 case 0452, Noir baseline.
// Kernel: `X / np.abs(X).sum(axis=1).reshape(-1, 1)` on Integer[5, 4].
// `/` is NumPy's true division, so each output is the Q32 quotient
// floor(x * 2^32 / l1) of two integers (positive-divisor floor division).
// |x| is noir_hw::abs (cheaper here than max_of([x, -x])). A zero row (l1 = 0,
// NumPy gives nan) has no proof. l1 gets a window check: with every |x| in
// [0, 2^63], it proves l1 < 2^63, an int64 as NumPy's sum is. So |i64::MIN| =
// 2^63, which overflows int64 (NumPy wraps it to i64::MIN), and a row sum past
// int64 have no proof, as in Zinnia; without the check the division by the
// unwrapped l1 proved a value NumPy does not give (X[0] = i64::MIN: -1.0 for
// NumPy's 1.0).
// @outputs: float[20]

use noir_hw::{abs, divmod_pos, window, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut l1: Field = 0;
        for j in 0..4 {
            l1 += abs(X[i * 4 + j]);
        }
        let _ = window(l1);
        for j in 0..4 {
            result[i * 4 + j] = divmod_pos(X[i * 4 + j] * SCALE, l1).0;
        }
    }
    result
}
