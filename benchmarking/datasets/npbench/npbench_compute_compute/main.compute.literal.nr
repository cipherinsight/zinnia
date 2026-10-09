// npbench_compute_compute__compute: literal translation of kernel.py (LITERAL_BASELINES.md).
// np.clip uses maximum then minimum; full array passes, exact integer arithmetic.
// @outputs: int[N * COLS]

use noir_hw::{check_input, check_inputs, max, min};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = array_1_0
global COLS: u32 = 16; // = array_1_1

fn main(array_1: [[Field; COLS]; N], array_2: [[Field; COLS]; N], a: Field, b: Field, c: Field) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is an int64.
    check_input(a);
    check_input(b);
    check_input(c);
    for i in 0..N { check_inputs(array_1[i]); check_inputs(array_2[i]); }
    let mut clipped = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { clipped[i][j] = min(max(array_1[i][j], 2), 10); } }
    let mut term1 = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { term1[i][j] = clipped[i][j] * a; } }
    let mut term2 = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { term2[i][j] = array_2[i][j] * b; } }
    let mut sums = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { sums[i][j] = term1[i][j] + term2[i][j]; } }
    let mut result = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { result[i][j] = sums[i][j] + c; } }
    result
}
