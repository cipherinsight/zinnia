// pythran_log_likelihood__log_likelihood: literal translation of kernel.py (LITERAL_BASELINES.md).
// Materialize each array; sqrt(2*pi) is a Python-level constant.
// Dynamic divisors use shared canonical_nonzero, never reused divisor proofs.

// @outputs: float[1]
use noir_hw::{canonical_div_nonzero, canonical_exp, canonical_log, check_input, check_inputs, mul};
// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = data_0
global SQRT_TWO_PI: Field = 10765886463;
fn main(data: [Field; N], mean: Field, sigma: Field) -> pub Field {
    check_inputs(data);
    check_input(mean);
    check_input(sigma);
    assert(sigma != 0);
    let mut deviations = [0; N];
    for i in 0..N {
        deviations[i] = data[i] - mean;
    }
    let mut squares = [0; N];
    for i in 0..N {
        squares[i] = mul(deviations[i], deviations[i]);
    }
    let sigma_sq = mul(sigma, sigma);
    let twice = 2 * sigma_sq;
    let mut s = [0; N];
    for i in 0..N {
        s[i] = canonical_div_nonzero(squares[i], twice);
    }
    let mut negative = [0; N];
    for i in 0..N {
        negative[i] = -s[i];
    }
    let mut pdfs = [0; N];
    for i in 0..N {
        pdfs[i] = canonical_exp(negative[i]);
    }
    let scale = mul(SQRT_TWO_PI, sigma);
    for i in 0..N {
        pdfs[i] = canonical_div_nonzero(pdfs[i], scale);
    }
    let mut logs = [0; N];
    for i in 0..N {
        logs[i] = canonical_log(pdfs[i]);
    }
    let mut total: Field = 0;
    for i in 0..N {
        total += logs[i];
    }
    total
}
