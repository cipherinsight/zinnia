// npbench_go_fast_go_fast__go_fast (NPBench go_fast), Noir baseline.
// Kernel: on a: Float[N, COLS], trace = sum_i np.tanh(a[i, i]) (from 0.0,
// over the diagonal; the loop runs over a.shape[0], so N <= COLS), then
// a + trace. `tanh` is the shared gadget (NPB_BASELINES.md section 5,
// rule 5); the sums are exact.
// @outputs: float[N * COLS]

use noir_hw::{check_inputs, tanh};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = a_0
global COLS: u32 = 16; // = a_1

fn main(a: [[Field; COLS]; N]) -> pub [[Field; COLS]; N] {
    // Input domain: every input element is a Q32 code in the window.
    for i in 0..N {
        check_inputs(a[i]);
    }
    let mut trace: Field = 0;
    for i in 0..N {
        trace += tanh(a[i][i]);
    }
    let mut result = a;
    for i in 0..N {
        for j in 0..COLS {
            result[i][j] += trace;
        }
    }
    result
}
