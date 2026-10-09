// npbench_polybench_durbin_durbin__kernel (NPBench polybench/durbin), Noir baseline.
// Kernel: Durbin's recursion on r: Float[N]: y[0] = alpha = -r[0],
// beta = 1.0; for k in 1..N: beta *= 1.0 - alpha * alpha,
// alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta,
// y[:k] += alpha * np.flip(y[:k]), y[k] = alpha; returns y.
// Rounding in NumPy's order: each product and the quotient is rounded, the dot
// product once, the negation applies before the division. The first
// beta *= ... multiplies by the constant 1.0, which is exact, so it is
// skipped; alpha * np.flip(y[:k]) reads y before the update.
// @outputs: float[N]

use noir_hw::{check_inputs, div, mul, rescale, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = r_0

fn main(r: [Field; N]) -> pub [Field; N] {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(r);
    let mut y: [Field; N] = [0; N];
    let mut alpha = -r[0];
    y[0] = alpha;
    let mut beta: Field = SCALE;
    for k in 1..N {
        let factor = SCALE - mul(alpha, alpha);
        beta = if k == 1 { factor } else { mul(beta, factor) };
        let mut acc: Field = 0;
        for j in 0..k {
            acc += r[k - 1 - j] * y[j];
        }
        alpha = div(-(r[k] + rescale(acc)), beta);
        let mut flipped: [Field; N] = [0; N];
        for j in 0..k {
            flipped[j] = mul(alpha, y[k - 1 - j]);
        }
        for j in 0..k {
            y[j] += flipped[j];
        }
        y[k] = alpha;
    }
    y
}
