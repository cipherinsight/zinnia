// pythran_multitype__check2: literal translation of kernel.py (LITERAL_BASELINES.md).
// Compute x+x in full, then select the tuple branch. Boolean array truth uses its sole element.
// @outputs: float[N]; bool[N]

use noir_hw::check_inputs;

// Sizes (hw_check.sizes): from fixture.check2.json.
global N: u32 = 1; // = x_0

fn main(x: [Field; N], y: [bool; N]) -> pub ([Field; N], [bool; N]) {
    check_inputs(x);
    assert(N == 1);
    let mut twice = [0; N];
    for i in 0..N {
        twice[i] = x[i] + x[i];
    }
    let mut result = [0; N];
    for i in 0..N {
        result[i] = if y[0] { x[i] } else { twice[i] };
    }
    (result, y)
}
