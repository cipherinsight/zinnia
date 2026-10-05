// DS-1000 case 0303, Noir baseline.
// Kernel: `np.reshape(A[:(23 // 5) * 5], (-1, 5))` on Float[23] (`ncol = 5` is
// a constant): the first 20 values of A as a (4, 5) array.
// @outputs: float[20]

fn main(A: [Field; 23]) -> pub [Field; 20] {
    let mut out = [0; 20];
    for k in 0..20 {
        out[k] = A[k];
    }
    out
}
