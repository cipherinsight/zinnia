// DS-1000 case 0485, Noir baseline.
// Kernel: `np.tensordot(A, B, axes=((2), (0)))` on A: Float[5, 6, 3], B: Float[3, 3]:
// r[i, j, k] = sum_p A[i, j, p] * B[p, k], each a dot product with one rescale.
// @outputs: float[90]

use noir_hw::dot;

fn main(A: [Field; 90], B: [Field; 9]) -> pub [Field; 90] {
    let mut r = [0; 90];
    for ij in 0..30 {
        for k in 0..3 {
            r[ij * 3 + k] = dot([A[ij * 3], A[ij * 3 + 1], A[ij * 3 + 2]], [B[k], B[3 + k], B[6 + k]]);
        }
    }
    r
}
