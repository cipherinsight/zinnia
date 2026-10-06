// DS-1000 case 0500, Noir baseline.
// Kernel: `a.reshape(-1, 3)` on Float[54]: the output Float[18, 3] is the input
// in the same row-major order.
// @outputs: float[54]

fn main(a: [Field; 54]) -> pub [Field; 54] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    a
}
