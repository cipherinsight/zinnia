// pythran_l2norm__l2_norm (Pythran l2norm), Noir baseline.
// Kernel: `np.sqrt(np.sum(np.abs(x) ** 2, 1))` on x: Float[N, COLS].
// np.abs(x) ** 2 is x ** 2, so the absolute value is skipped; each row's sum
// of individually rounded squares feeds the square root.
// @outputs: float[N]

use noir_hw::{check_inputs, mul, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0
global COLS: u32 = 16; // = x_1

fn main(x: [[Field; COLS]; N]) -> pub [Field; N] {
    let mut norms: [Field; N] = [0; N];
    for i in 0..N {
        // Input domain: every input element is a Q32 code in the window.
        check_inputs(x[i]);
        let mut total: Field = 0;
        for k in 0..COLS { total += mul(x[i][k], x[i][k]); }
        norms[i] = sqrt(total);
    }
    norms
}
