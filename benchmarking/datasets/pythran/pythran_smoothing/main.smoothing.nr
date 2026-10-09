// pythran_smoothing__smoothing (Pythran smoothing), Noir baseline.
// Kernel: exponential smoothing of x: Float[N] with a scalar alpha:
// s[0] = x[0], s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]; returns s.
// Both products of a step are rounded (scalar operations), 1 - alpha is exact
// and computed once.
// @outputs: float[N]

use noir_hw::{check_input, check_inputs, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N], alpha: Field) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    check_input(alpha);
    let keep = SCALE - alpha;
    let mut s = x;
    for i in 1..N {
        s[i] = mul(alpha, x[i]) + mul(keep, s[i - 1]);
    }
    s
}
