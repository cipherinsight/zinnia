// DS-1000 case 0327, Noir baseline.
// Kernel: `np.pad(A, (0, 17 - 10), 'constant')` on Float[10] (`length = 17` is
// a constant): A followed by seven zeros, which are circuit constants.
// @outputs: float[10]; const float[7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

fn main(A: [Field; 10]) -> pub [Field; 10] {
    A
}
