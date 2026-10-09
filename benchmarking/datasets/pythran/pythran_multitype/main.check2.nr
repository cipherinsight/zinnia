// pythran_multitype__check2 (Pythran multitype), Noir baseline.
// Kernel: `(x, y) if y else (x + x, y)` on x: Float[N] and y: Boolean[N]
// (N = 1: `if y` takes the truth value of a one-element array); the tuple is
// output in order.
// The branch is a select on the bit (a `bool` parameter is constrained to
// 0 or 1 by the ABI); y is output as loaded.
// @outputs: float[N]; bool[N]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.check2.json.
global N: u32 = 1; // = x_0

fn main(x: [Field; N], y: [bool; N]) -> pub ([Field; N], [bool; N]) {
    assert(N == 1, "`if y` needs a one-element array");
    // Input domain: x is a Q32 code in the window.
    check_inputs(x);
    let falsy = (!y[0]) as Field;
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        result[i] = x[i] + falsy * x[i];
    }
    (result, y)
}
