// DS-1000 case 0301, Noir baseline.
// Kernel: `np.reshape(A, (-1, 5))` on Float[20] (`ncol = 5` is a constant): a
// (4, 5) view of A in the same row-major order, so the output is the input.
// @outputs: float[20]

fn main(A: [Field; 20]) -> pub [Field; 20] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(A);
    A
}
