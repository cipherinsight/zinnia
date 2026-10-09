// pythran_log_likelihood__log_likelihood (Pythran log_likelihood), Noir baseline.
// Kernel: on data: Float[N], mean, sigma: Float, with the precondition
// sigma != 0:
//     s = (data - mean) ** 2 / (2 * sigma ** 2)
//     pdfs = numpy.exp(-s)
//     pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
//     return numpy.log(pdfs).sum()
// Stable algebraic evaluation: log(exp(-s) / scale) = -s - log(scale).
// Hoist log(sqrt(2 pi) * sigma) out of the reduction. This avoids Q32 pdf
// underflow and saves per-element exp, log and division. Squares and s
// retain Q32 rounding; the identity precedes materializing the pdf in Q32.
// log(scale) requires positive sigma, necessary for finite NumPy densities.
// @outputs: float[1]

use noir_hw::{check_input, check_inputs, divmod_pos, log, mul, SCALE};

// Sizes (hw_check.sizes): from fixture.json.
global N: u32 = 16; // = data_0
global SQRT_TWO_PI: Field = 10765886463; // round(sqrt(2 pi) * 2^32)

fn main(data: [Field; N], mean: Field, sigma: Field) -> pub Field {
    // Input domain: every input element is a Q32 code in the window.
    check_inputs(data);
    check_input(mean);
    check_input(sigma);
    let twice = 2 * mul(sigma, sigma);
    let scale = mul(SQRT_TWO_PI, sigma);
    let normalization = log(scale);
    let mut total: Field = 0;
    for k in 0..N {
        let deviation = data[k] - mean;
        let (s, _) = divmod_pos(mul(deviation, deviation) * SCALE, twice);
        total -= s + normalization;
    }
    total
}
