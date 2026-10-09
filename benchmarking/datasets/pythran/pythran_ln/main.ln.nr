// pythran_ln__ln (Pythran ln), Noir baseline.
// Kernel: Y[:] = d - d ** 2 / 2 + d ** 3 / 3 - ... + d ** 9 / 9 with
// d = X - 1 on X, Y: Float[N]; returns Y. Y is overwritten before it is read,
// so it is not checked.
// Each power follows MSB-first square-and-multiply; only identical
// intermediate products are shared. Constant positive integer division
// is floor division of the Q32 code, preserving the source rounding.
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
        let d6 = mul(d3, d3);
        let d7 = mul(d6, d);
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
