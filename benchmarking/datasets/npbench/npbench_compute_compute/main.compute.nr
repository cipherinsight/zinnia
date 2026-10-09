// npbench_compute_compute__compute (NPBench compute), Noir baseline.
// Kernel: np.clip(array_1, 2, 10) * a + array_2 * b + c on int64 arrays
// array_1, array_2: Integer[N, COLS] and int64 scalars a, b, c.
// The clip is `clip_const` (region bits pinned by one 64-bit range check) on
// each array_1 element, which its input check puts in the window; the rest is
// exact Field arithmetic.
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
            result[i][j] = clip_const(array_1[i][j], 2, 10) * a + array_2[i][j] * b + c;
        }
    }
    result
}
