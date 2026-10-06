// DS-1000 case 0505, Noir baseline.
// Kernel: `a.sum() - np.take_along_axis(a, b[..., None], axis=-1)[..., 0].sum()` on a: Float[13, 13, 5], b: Integer[13, 13].
// Each b[i, j] selects among the 5 values a[i, j, :] with NumPy's index rules
// (negative indices wrap), by a one-hot over the valid index values (cheaper
// than native indexing for rows this short).
// @outputs: float[1]

use noir_hw::pick;

fn main(a: [Field; 845], b: [Field; 169]) -> pub Field {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(b);
    let mut all: Field = 0;
    for i in 0..845 {
        all += a[i];
    }
    let mut taken: Field = 0;
    for ij in 0..169 {
        taken += pick([a[ij * 5 + 0], a[ij * 5 + 1], a[ij * 5 + 2], a[ij * 5 + 3], a[ij * 5 + 4]], b[ij]);
    }
    all - taken
}
