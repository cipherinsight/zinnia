//! Pythran train_eq: training loop of a blind adaptive equalizer (constant
//! modulus algorithm) over a window that slides along the received symbols.
//! The two entry functions are the same text.
//!
//! ```python
//! def train_eq(E: Complex[Array, '1 100'], TrSyms: Integer, os: Integer, mu: Float, wx: Complex[Array, '1 10'], errfctprs: Tuple[Complex, Complex[Array, '4']], adapt: Boolean):
//!     Ntaps = wx.shape[1]
//!     pols = wx.shape[0]
//!     R, symbs = errfctprs
//!     err = np.zeros(TrSyms, dtype=E.dtype)
//!     for i in range(TrSyms):
//!         X = E[:, i * os:i * os + Ntaps]
//!         Xest = np.sum(np.conj(wx) * X)
//!         err[i] = (R.real - abs(Xest) ** 2) * Xest
//!         wx += mu * np.conj(err[i]) * X
//!         if adapt and i > 0:
//!             if err[i].real * err[i - 1].real > 0 and err[i].imag * err[i - 1].imag > 0:
//!                 pass
//!             else:
//!                 mu = np.float32(mu / (1 + mu * (err[i].real * err[i].real + err[i].imag * err[i].imag)))
//!     return (err, wx, mu)
//!
//! def train_eq2(...):
//!     (the same body)
//! ```
use core::ops::Range;

use crate::prelude::*;

/// The indices that the slice `start:stop` selects on an axis of length
/// `len` (Python slicing: negative bounds count from the end, bounds are
/// clipped to the axis, and `stop <= start` selects nothing).
fn py_slice(start: i64, stop: i64, len: usize) -> Range<usize> {
    let n = len as i64;
    let clip = |i: i64| (if i < 0 { (i + n).max(0) } else { i.min(n) }) as usize;
    let (start, stop) = (clip(start), clip(stop));
    start..stop.max(start)
}

/// `np.float32(x)`: `x` rounded to the nearest float32, i.e. to 24
/// significant bits, ties to even. Every nonzero Q32 value lies in float32's
/// normal range, so only the significand is rounded.
fn round_to_f32(x: Q32) -> Q32 {
    let magnitude = x.to_bits().unsigned_abs();
    let width = u64::BITS - magnitude.leading_zeros();
    if width <= 24 {
        return x;
    }
    let shift = width - 24;
    let (kept, dropped) = (magnitude >> shift, magnitude & ((1 << shift) - 1));
    let half = 1 << (shift - 1);
    let kept = if dropped > half || (dropped == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    };
    let rounded = (kept << shift) as i64;
    Q32::from_bits(if x.is_negative() { -rounded } else { rounded })
}

/// Both entry functions: reads the parameters and runs the training loop.
fn train(args: &mut Args<impl Input>) -> (Array1<C32>, Array2<C32>, Q32) {
    let e: Array2<C32> = args.array2();
    let tr_syms: i64 = args.scalar();
    let os: i64 = args.scalar();
    let mut mu: Q32 = args.scalar();
    let mut wx: Array2<C32> = args.array2();
    // errfctprs = (R, symbs) arrives as its two items in order; symbs is
    // never used.
    let r: C32 = args.scalar();
    let _symbs: Array1<C32> = args.array1();
    let adapt: bool = args.scalar();

    let ntaps = wx.ncols() as i64;
    let mut err = Array1::<C32>::zeros(
        usize::try_from(tr_syms).expect("negative dimensions are not allowed"),
    );
    for i in 0..err.len() {
        let start = i as i64 * os;
        let window = e.slice(s![.., py_slice(start, start + ntaps, e.ncols())]);
        // `wx += ... * X` needs X to broadcast to wx's shape.
        let x = window
            .broadcast(wx.dim())
            .expect("operands could not be broadcast together");
        let xest: C32 = Zip::from(&wx)
            .and(&x)
            .fold(C32::ZERO, |sum, &w, &v| sum + w.conj() * v);
        let magnitude = xest.abs();
        let err_i = xest * (r.re - magnitude * magnitude);
        err[i] = err_i;
        let step = err_i.conj() * mu;
        Zip::from(&mut wx).and(&x).for_each(|w, &v| *w += step * v);
        if adapt && i > 0 {
            let prev = err[i - 1];
            let same_signs = err_i.re * prev.re > Q32::ZERO && err_i.im * prev.im > Q32::ZERO;
            if !same_signs {
                let power = err_i.re * err_i.re + err_i.im * err_i.im;
                mu = round_to_f32(mu / (Q32::ONE + mu * power));
            }
        }
    }
    (err, wx, mu)
}

pub mod train_eq {
    use super::*;

    pub type Output = (Array1<C32>, Array2<C32>, Q32);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        train(args)
    }
}

pub mod train_eq2 {
    use super::*;

    pub type Output = (Array1<C32>, Array2<C32>, Q32);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        train(args)
    }
}
