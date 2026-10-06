// DS-1000 case 0340, Noir baseline.
// Kernel: `[value for value in X.flat]` on Integer[5, 6]: the 30 elements in
// row-major order.
// @outputs: int[30]

fn main(X: [Field; 30]) -> pub [Field; 30] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(X);
    X
}
