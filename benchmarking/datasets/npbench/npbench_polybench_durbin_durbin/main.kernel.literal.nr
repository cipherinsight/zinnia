// npbench_polybench_durbin_durbin__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// Each dot rescales once; the slice product is fully materialized before updating y. No first-iteration shortcut.
// @outputs: float[N]

use noir_hw::{check_inputs, div, mul, rescale, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = r_0

fn main(r: [Field; N]) -> pub [Field; N] {
    check_inputs(r);
    let mut y = [0; N];
    let mut alpha = -r[0];
    let mut beta = SCALE;
    y[0] = -r[0];
    for k in 1..N {
        beta = mul(beta, SCALE - mul(alpha, alpha));
        let mut acc = 0;
        for j in 0..k { acc += r[k-1-j] * y[j]; }
        alpha = div(-(r[k] + rescale(acc)), beta);
        let mut products = [0; N];
        for j in 0..k { products[j] = mul(alpha, y[k-1-j]); }
        for j in 0..k { y[j] += products[j]; }
        y[k] = alpha;
    }
    y
}
