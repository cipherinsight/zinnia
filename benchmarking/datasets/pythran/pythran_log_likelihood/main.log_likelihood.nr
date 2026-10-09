// Literal square/divide/exp/density-divide/log/sum sequence from kernel.py.
// Canonical Q32; no log(exp(x)) identity or log-normalization hoisting.
// @outputs: float[1]
use noir_hw::{check_input, check_inputs, div, exp, log, mul};
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = data_0
global SQRT_TWO_PI: Field = 10765886463;
fn main(data: [Field; N], mean: Field, sigma: Field) -> pub Field {
    check_inputs(data);
    check_input(mean);
    check_input(sigma);
    let twice = 2 * mul(sigma, sigma);
    let scale = mul(SQRT_TWO_PI, sigma);
    let mut total: Field = 0;
    for k in 0..N {
        let deviation = data[k] - mean;
        let s = div(mul(deviation, deviation), twice);
        let density = exp(-s);
        let normalized = div(density, scale);
        total += log(normalized);
    }
    total
}
