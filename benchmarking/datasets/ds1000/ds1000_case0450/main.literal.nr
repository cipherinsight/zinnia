// DS-1000 case 0450, Noir literal-translation baseline (LITERAL_BASELINES.md).
// Kernel: `result = a[:, np.array(second).reshape(-1, 1), third]` on
// Float[7, 8, 9] with second, third Integer[2]: NumPy broadcasts second
// (reshaped to (2, 1)) against third ((2,)) over axes 1 and 2, so
// result[i, j, k] = a[i, second[j], third[k]], shape (7, 2, 2). Negative
// indices wrap (NumPy rule); any other out-of-range index raises
// IndexError (no satisfying witness).
//
// Translation choices:
// - The one statement reads two private indices at once, one per axis.
//   Translated axis by axis in the order written, `a[..., second[j],
//   ...]` then `[..., third[k]]`: for each i, j, gather the row
//   `a[i, second[j], :]` (index_pos on axis 1, 9 native reads), then read
//   third[k] out of that row (index_pos on axis 2). Nothing is cached
//   across i or j: each index_pos call is fresh, as the source has no
//   caching across the broadcast (the optimized baseline resolves each
//   index once and reuses it for every i, which its own header names as
//   the optimization this baseline does not take).
// @outputs: float[28]

use noir_hw::index_pos;

global D1: u32 = 8;
global D2: u32 = 9;

fn main(a: [Field; 7 * D1 * D2], second: [Field; 2], third: [Field; 2]) -> pub [Field; 28] {
    // Input domain: every input element is an int64 / a Q32 code in the window.
    noir_hw::check_inputs(a);
    noir_hw::check_inputs(second);
    noir_hw::check_inputs(third);

    let mut out: [Field; 28] = [0; 28];
    for i in 0..7 {
        for j in 0..2 {
            // row = a[i, second[j], :]
            let p = index_pos::<8>(second[j], D1);
            let mut row: [Field; 9] = [0; 9];
            for p2 in 0..9 {
                row[p2] = a[i * D1 * D2 + p * D2 + p2];
            }
            for k in 0..2 {
                let qpos = index_pos::<8>(third[k], D2);
                out[i * 4 + j * 2 + k] = row[qpos];
            }
        }
    }
    out
}
