// DS-1000 case 0504, Noir baseline.
// Kernel: `np.take_along_axis(a, b[..., None], axis=-1)[..., 0].sum()` on a: Float[13, 13, 5], b: Integer[13, 13].
// Each b[i, j] selects among the 5 values a[i, j, :] with NumPy's index rules
// (negative indices wrap), by a one-hot over the valid index values (cheaper
// than native indexing for rows this short).
// @outputs: float[1]

use noir_hw::pick;

fn main(a: [Field; 845], b: [Field; 169]) -> pub Field {
    let mut total: Field = 0;
    for ij in 0..169 {
        total += pick([a[ij * 5 + 0], a[ij * 5 + 1], a[ij * 5 + 2], a[ij * 5 + 3], a[ij * 5 + 4]], b[ij]);
    }
    total
}
