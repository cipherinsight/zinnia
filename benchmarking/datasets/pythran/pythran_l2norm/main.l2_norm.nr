// pythran_l2norm__l2_norm (Pythran l2norm), Noir baseline.
// Kernel: `np.sqrt(np.sum(np.abs(x) ** 2, 1))` on x: Float[N, COLS].
// np.abs(x) ** 2 is x ** 2, so the absolute value is skipped; each row's sum
// of squares is one reduction, rescaled once (NPB_BASELINES.md section 5,
// rule 3), and its square root is the exact floor square root.
// @outputs: float[N]

use noir_hw::{check_inputs, dot, sqrt};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = x_0
global COLS: u32 = 16; // = x_1

fn main(x: [[Field; COLS]; N]) -> pub [Field; N] {
    let mut norms: [Field; N] = [0; N];
    for i in 0..N {
        // Input domain: every input element is a Q32 code in the window.
        check_inputs(x[i]);
        norms[i] = sqrt(dot(x[i], x[i]));
    }
    norms
}
