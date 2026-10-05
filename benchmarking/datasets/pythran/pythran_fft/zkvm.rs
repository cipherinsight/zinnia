//! Pythran fft: the recursive radix-2 Cooley-Tukey FFT.
//!
//! ```python
//! def fft(x: Complex[Array, '1024']):
//!     N = x.shape[0]
//!     if N == 1:
//!         return np.array(x)
//!     e = fft(x[::2])
//!     o = fft(x[1::2])
//!     M = N // 2
//!     l = [e[k] + o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
//!     r = [e[k] - o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
//!     return np.array(l + r)
//! ```
use crate::prelude::*;

fn fft(x: ArrayView1<C32>) -> Array1<C32> {
    let n = x.len();
    // For N = 0, `x[::2]` is empty again and the recursion never ends.
    assert!(n != 0, "maximum recursion depth exceeded");
    if n == 1 {
        return x.to_owned();
    }
    let e = fft(x.slice(s![..;2]));
    let o = fft(x.slice(s![1..;2]));
    let m = n / 2;
    let mut out = Array1::zeros(2 * m);
    for k in 0..m {
        // `math.e ** (-2j * math.pi * k / N)` = exp(i * (-2 pi k / N)).
        let twiddle = C32::new(Q32::ZERO, Q32::PI * -2 * k as i64 / n as i64).exp();
        let t = o[k] * twiddle;
        out[k] = e[k] + t;
        out[k + m] = e[k] - t;
    }
    out
}

pub mod fft {
    use super::*;

    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let x: Array1<C32> = args.array1();
        super::fft(x.view())
    }
}
