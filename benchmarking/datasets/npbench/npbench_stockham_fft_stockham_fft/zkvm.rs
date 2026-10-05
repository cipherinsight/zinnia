//! NPBench stockham_fft: radix-`R` Stockham FFT of `N = R**K` points, in
//! place in `y`.
//!
//! ```python
//! @requires(lambda N, R, K, x, y: R != 0)
//! def stockham_fft(N: Integer, R: Integer, K: Integer, x: Complex[Array, '32768'], y: Complex[Array, '32768']):
//!     i_coord, j_coord = np.mgrid[0:R, 0:R]
//!     dft_mat = np.empty((R, R), dtype=np.complex128)
//!     dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
//!     y[:] = x[:]
//!     ii_coord, jj_coord = np.mgrid[0:R, 0:R ** K]
//!     for i in range(K):
//!         yv = np.reshape(y, (R ** i, R, R ** (K - i - 1)))
//!         tmp_perm = np.transpose(yv, axes=(1, 0, 2))
//!         D = np.empty((R, R ** i, R ** (K - i - 1)), dtype=np.complex128)
//!         tmp = np.exp(-2j * np.pi * ii_coord[:, :R ** i] * jj_coord[:, :R ** i] / R ** (i + 1))
//!         D[:] = np.repeat(np.reshape(tmp, (R, R ** i, 1)), R ** (K - i - 1), axis=2)
//!         tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
//!         y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))
//! ```
use crate::prelude::*;

/// `np.exp(-2j * np.pi * a * b / d)`. The product `-2j * pi * a * b` is
/// exact in Q32 (its real part is 0, so the exponential is `cos + i sin` of
/// the angle).
fn twiddle(a: usize, b: usize, d: i64) -> C32 {
    let angle = Q32::PI * (-2 * (a * b) as i64) / d;
    C32::new(angle.cos(), angle.sin())
}

/// `a @ b` for complex vectors: each part accumulates its raw products and
/// rescales once, like `np::inner`.
fn complex_inner(a: ArrayView1<C32>, b: ArrayView1<C32>) -> C32 {
    let pairs = || a.iter().zip(b.iter());
    C32::new(
        dot(pairs().flat_map(|(x, y)| [(x.re, y.re), (-x.im, y.im)])),
        dot(pairs().flat_map(|(x, y)| [(x.re, y.im), (x.im, y.re)])),
    )
}

pub mod stockham_fft {
    use super::*;

    // outputs: mutated y
    pub type Output = Array1<C32>;

    #[allow(non_snake_case)]
    pub fn main(args: &mut Args<impl Input>) -> Output {
        let N: i64 = args.scalar();
        let R: i64 = args.scalar();
        let K: i64 = args.scalar();
        let x: Array1<C32> = args.array1();
        let mut y: Array1<C32> = args.array1();
        assert!(R != 0, "requires R != 0");
        // `y[:] = x[:]` (broadcasting like NumPy).
        y.assign(&x);
        if K <= 0 {
            return y;
        }
        let r = usize::try_from(R).expect("negative dimensions are not allowed");
        let k = K as u32;
        let n = r.pow(k);
        assert!(
            y.len() == n && N == n as i64,
            "cannot reshape y to (R ** i, R, R ** (K - i - 1)) or (N,)"
        );
        let dft = Array2::from_shape_fn((r, r), |(a, b)| twiddle(a, b, R));
        let cols = n / r; // R ** (K - 1)
        for i in 0..k {
            let (outer, inner) = (r.pow(i), r.pow(k - i - 1)); // R ** i, R ** (K - i - 1)
            let tmp = Array2::from_shape_fn((r, outer), |(s, a)| twiddle(s, a, R.pow(i + 1)));
            // tmp_twid[s, a, b] = yv[a, s, b] * D[s, a, b], viewed as R x R ** (K - 1).
            let twiddled = Array2::from_shape_fn((r, cols), |(s, c)| {
                let (a, b) = (c / inner, c % inner);
                y[(a * r + s) * inner + b] * tmp[(s, a)]
            });
            y = Array1::from_shape_fn(n, |m| {
                let (row, c) = (m / cols, m % cols);
                complex_inner(dft.row(row), twiddled.column(c))
            });
        }
        y
    }
}
