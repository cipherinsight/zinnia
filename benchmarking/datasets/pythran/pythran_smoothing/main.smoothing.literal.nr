// pythran_smoothing__smoothing: literal translation of kernel.py (LITERAL_BASELINES.md).
// Recompute 1-alpha inside each recurrence, in source evaluation order.

// @outputs: float[N]

use noir_hw::{check_input, check_inputs, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0

fn main(x: [Field; N], alpha: Field) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(x);
    check_input(alpha);
    let mut s = x;
    for i in 1..N {
        let fresh = mul(alpha, x[i]);
        let keep = SCALE - alpha;
        let carried = mul(keep, s[i - 1]);
        s[i] = fresh + carried;
    }
    s
}
