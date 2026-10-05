// DS-1000 case 0304, Noir baseline.
// Kernel: `np.reshape(A[23 - 20:][::-1], (-1, 5))` on Float[23] (`ncol = 5` is
// a constant, so 20 values are kept): A[22], A[21], ..., A[3] as a (4, 5) array.
// @outputs: float[20]

fn main(A: [Field; 23]) -> pub [Field; 20] {
    let mut out = [0; 20];
    for k in 0..20 {
        out[k] = A[22 - k];
    }
    out
}
