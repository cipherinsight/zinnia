// DS-1000 case 0302, Noir baseline.
// Kernel: `np.reshape(A, (5, -1))` on Float[20] (`nrow = 5` is a constant): a
// (5, 4) view of A in the same row-major order, so the output is the input.
// @outputs: float[20]

fn main(A: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(A);
    A
}
