// pythran_ln__ln (Pythran ln), Noir baseline.
// Kernel: Y[:] = d - d ** 2 / 2 + d ** 3 / 3 - ... + d ** 9 / 9 with
// d = X - 1 on X, Y: Float[N]; returns Y. Y is overwritten before it is read,
// so it is not checked.
// Each power is the square-and-multiply product chain of the halo2 baseline's
// `q.pow_const`; the chains of the eight powers share their squares and
// d ** 3, so the eight powers cost eight rounded products. v / n by the
// integer constant n is floor(v * 2^32 / (n * 2^32)) = floor(v / n):
// `floor_div_pow2` for n = 2, 4, 8 (a remainder of 1 to 3 bits; 4,869 gates,
// against 4,971 with `floor_div_const` for every n) and `floor_div_const::<n>`
// for the others, whose remainder checks are against n itself.
// @outputs: float[N]

use noir_hw::{check_inputs, floor_div_const, floor_div_pow2, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = X_0

fn main(X: [Field; N], Y: [Field; N]) -> pub [Field; N] {
    // Input domain: every element of X is a Q32 code in the window.
    check_inputs(X);
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        let d = X[i] - SCALE;
        let d2 = mul(d, d);
        let d4 = mul(d2, d2);
        let d8 = mul(d4, d4);
        let d3 = mul(d, d2);
        let d5 = mul(d, d4);
        let d6 = mul(d2, d4);
        let d7 = mul(d3, d4);
        let d9 = mul(d, d8);
        result[i] = d - floor_div_pow2::<1>(d2) + floor_div_const::<3>(d3) - floor_div_pow2::<2>(d4)
            + floor_div_const::<5>(d5)
            - floor_div_const::<6>(d6)
            + floor_div_const::<7>(d7)
            - floor_div_pow2::<3>(d8)
            + floor_div_const::<9>(d9);
    }
    result
}
