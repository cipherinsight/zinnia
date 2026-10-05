//! DS-1000 case 0422: circular cross-correlation via
//! `np.correlate(a, np.hstack((b[1:], b)), mode='valid')`.
//!
//!     def kernel(a: Float[Array, "50"], b: Float[Array, "50"]):
//!         result = np.correlate(a, np.hstack((b[1:], b)), mode='valid')
//!         return result
// output: [q32]
use crate::q32::{Q32, from_raw_product, raw_product};

const N: u32 = 50;

#[executable]
fn main(a: Array<Q32>, b: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N && b.len() == N, "inputs have the wrong length");
    let mut v = array![];
    v.append_span(b.span().slice(1, N - 1));
    v.append_span(b.span());
    correlate_valid_longer_v(a.span(), v.span())
}

/// `np.correlate(a, v, mode='valid')` for real `v` longer than `a`. NumPy then
/// slides `a` along `v` and reverses the result:
/// `c[k] = sum_n a[n] * v[n + len(v) - len(a) - k]`, i.e. the windows of `v`
/// from the last to the first, each a dot product accumulated unscaled and
/// rescaled once.
fn correlate_valid_longer_v(a: Span<Q32>, v: Span<Q32>) -> Array<Q32> {
    let lags = v.len() - a.len() + 1;
    let mut out = array![];
    for k in 0..lags {
        let window = v.slice(lags - 1 - k, a.len());
        let mut acc: i128 = 0;
        for n in 0..a.len() {
            acc += raw_product(*a.at(n), *window.at(n));
        }
        out.append(from_raw_product(acc));
    }
    out
}
