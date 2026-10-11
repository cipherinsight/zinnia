// DS-1000 case 0453, Noir baseline.
// Kernel: `X / np.sqrt((X*X).sum(axis=-1)).reshape(-1, 1)` on Integer[5, 4].
// The squares and their row sum stay integers; the norm is the Q32 square root
// of that sum, and `/` is NumPy's true division (a Q32 float division by the
// non-negative norm). No proof for a zero row or sum(X*X) >= 2^32:
// the compiled Noir circuit checks the Q32 radicand as an unsigned 64-bit
// value (unlike Halo2's signed64 check, which requires sum < 2^31).
// Checked int64 inputs make the four-square sum exact and at most 2^128,
// far below the field modulus even after Q32 scaling. Thus one 32-bit
// row check enforces the domain and implies the old per-element bounds.
// The division constraints already reject a zero norm.
// @outputs: float[20]

use noir_hw::{divmod_pos, from_int, sqrt, SCALE};

fn main(X: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    let mut result: [Field; 20] = [0; 20];
    for i in 0..5 {
        let mut sq_sum: Field = 0;
        for j in 0..4 {
            sq_sum += X[i * 4 + j] * X[i * 4 + j];
        }
        sq_sum.assert_max_bit_size::<32>();
        let l2 = sqrt(from_int(sq_sum));
        for j in 0..4 {
            // floor(float(x) * 2^32 / l2)
            result[i * 4 + j] = divmod_pos(from_int(X[i * 4 + j]) * SCALE, l2).0;
        }
    }
    result
}
