//! Pythran log_likelihood: the log-likelihood of `data` under the normal
//! distribution `N(mean, sigma^2)`.
//!
//! ```python
//! def log_likelihood(data: Float[Array, '10000'], mean: Float, sigma: Float):
//!     s = (data - mean) ** 2 / (2 * sigma ** 2)
//!     pdfs = numpy.exp(-s)
//!     pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
//!     return numpy.log(pdfs).sum()
//! ```
use crate::prelude::*;

/// `numpy.sqrt(2 * numpy.pi)`, quantized.
const SQRT_2PI: Q32 = Q32::from_bits(10_765_886_463);

pub mod log_likelihood {
    use super::*;

    pub type Output = Q32;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let data: Array1<Q32> = args.array1();
        let mean: Q32 = args.scalar();
        let sigma: Q32 = args.scalar();
        assert!(sigma != Q32::ZERO, "requires sigma != 0.0");
        let two_var = sigma * sigma * 2;
        let normalizer = SQRT_2PI * sigma;
        data.iter()
            .map(|&x| {
                let s = (x - mean) * (x - mean) / two_var;
                ((-s).exp() / normalizer).ln()
            })
            .sum()
    }
}
