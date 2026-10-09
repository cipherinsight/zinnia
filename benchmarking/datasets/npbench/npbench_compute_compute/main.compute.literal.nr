// npbench_compute_compute__compute (NPBench compute), Noir literal baseline.
// LITERAL_BASELINES.md: direct translation of kernel.py, no optimization.
//
// Source (one statement, kernel.py):
//   return np.clip(array_1, 2, 10) * a + array_2 * b + c
// Translation choices:
// - `np.clip` on an int64 array (L5 table): `clip_const`, the shared
//   region-pinning gadget for a clip against compile-time constant bounds.
// - All values are `Integer` (not `Number`): plain Field arithmetic, no
//   rescale (there is no fixed-point scale to divide out).
// - The expression is evaluated in the same left-to-right grouping Python
//   uses: `((clip(array_1[i][j]) * a) + (array_2[i][j] * b)) + c`.
// @outputs: int[N * COLS]

use noir_hw::{check_input, check_inputs, clip_const};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = array_1_0
global COLS: u32 = 16; // = array_1_1

fn main(array_1: [[Field; COLS]; N], array_2: [[Field; COLS]; N], a: Field, b: Field, c: Field) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is an int64.
    check_input(a);
    check_input(b);
    check_input(c);
    let mut result: [[Field; COLS]; N] = [[0; COLS]; N];
    for i in 0..N {
        check_inputs(array_1[i]);
        check_inputs(array_2[i]);
        for j in 0..COLS {
            let clipped = clip_const(array_1[i][j], 2, 10);
            result[i][j] = (clipped * a) + (array_2[i][j] * b) + c;
        }
    }
    result
}
