// DS-1000 case 0328, Noir baseline.
// Kernel: `result = A.copy(); result[3:] = 0` on Float[5] (`length = 3` is a
// constant and not above A.shape[0], so the np.pad branch is dead): the first
// three values of A, then two zeros, which are circuit constants.
// @outputs: float[3]; const float[2] = [0.0, 0.0]

fn main(A: [Field; 5]) -> pub [Field; 3] {
    [A[0], A[1], A[2]]
}
