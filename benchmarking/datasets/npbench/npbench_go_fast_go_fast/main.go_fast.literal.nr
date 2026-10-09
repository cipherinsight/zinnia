// npbench_go_fast_go_fast__go_fast: literal translation of kernel.py (LITERAL_BASELINES.md).
// CHECKER BLOCKER: Halo2 L5/L6 per-operation input bounds conflict with the checker's
// zero redundant_input_checks requirement; see LITERAL_NPB3.md. No diagnostic bypass.
// Sequential trace recurrence; canonical shared tanh matches the optimized arithmetic.
// @outputs: float[N * COLS]

use noir_hw::{check_inputs, canonical_tanh as tanh};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = a_0
global COLS: u32 = 16; // = a_1

fn main(a: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(a[i]);
    }
    let mut trace: Field = 0;
    for i in 0..N { trace += tanh(a[i][i]); }
    let mut result = [[0; COLS]; N];
    for i in 0..N { for j in 0..COLS { result[i][j] = a[i][j]+trace; } }
    result
}
