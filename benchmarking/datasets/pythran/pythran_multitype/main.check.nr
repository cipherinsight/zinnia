// pythran_multitype__check (Pythran multitype), Noir baseline.
// Kernel: `(x, y) if y else (x + x, y)` on x: Float[N] (N = 1) and a private
// integer y; the tuple is output in order.
// The branch is a select on y == 0; y is output as loaded.
// @outputs: float[N]; int[1]

use noir_hw::{check_input, check_inputs};

// Sizes (hw_check.sizes): from fixture.check.json.
global N: u32 = 1; // = x_0

fn main(x: [Field; N], y: Field) -> pub ([Field; N], Field) {
    // Input domain: x is a Q32 code and y an int64, in the window.
    check_inputs(x);
    check_input(y);
    let falsy = (y == 0) as Field;
    let mut result: [Field; N] = [0; N];
    for i in 0..N {
        result[i] = x[i] + falsy * x[i];
    }
    (result, y)
}
