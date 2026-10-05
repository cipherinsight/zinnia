// DS-1000 case 0502, Noir baseline.
// Kernel: `np.take_along_axis(a, b[..., None], axis=-1)[..., 0]` on a: Float[13, 13, 2], b: Integer[13, 13].
// Each b[i, j] selects among the 2 values a[i, j, :] with NumPy's index rules
// (negative indices wrap), by a one-hot over the valid index values (cheaper
// than native indexing for rows this short).
// @outputs: float[169]

use noir_hw::pick;

fn main(a: [Field; 338], b: [Field; 169]) -> pub [Field; 169] {
    let mut picked = [0; 169];
    for ij in 0..169 {
        picked[ij] = pick([a[ij * 2 + 0], a[ij * 2 + 1]], b[ij]);
    }
    picked
}
