//! NPBench polybench/deriche: the Deriche recursive edge-detection filter,
//! a causal and an anti-causal pass along the rows, then along the columns.
//!
//! ```python
//! def kernel(alpha: Float, imgIn: Float[Array, '400 200']):
//!     k = (1.0 - np.exp(-alpha)) * (1.0 - np.exp(-alpha)) / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
//!     a1 = a5 = k
//!     a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
//!     a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
//!     a4 = a8 = -k * np.exp(-2.0 * alpha)
//!     b1 = 2.0 ** (-alpha)
//!     b2 = -np.exp(-2.0 * alpha)
//!     c1 = c2 = 1
//!     y1 = np.empty_like(imgIn)
//!     y1[:, 0] = a1 * imgIn[:, 0]
//!     y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
//!     for j in range(2, imgIn.shape[1]):
//!         y1[:, j] = a1 * imgIn[:, j] + a2 * imgIn[:, j - 1] + b1 * y1[:, j - 1] + b2 * y1[:, j - 2]
//!     y2 = np.empty_like(imgIn)
//!     y2[:, -1] = 0.0
//!     y2[:, -2] = a3 * imgIn[:, -1]
//!     for j in range(imgIn.shape[1] - 3, -1, -1):
//!         y2[:, j] = a3 * imgIn[:, j + 1] + a4 * imgIn[:, j + 2] + b1 * y2[:, j + 1] + b2 * y2[:, j + 2]
//!     imgOut = c1 * (y1 + y2)
//!     y1[0, :] = a5 * imgOut[0, :]
//!     y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
//!     for i in range(2, imgIn.shape[0]):
//!         y1[i, :] = a5 * imgOut[i, :] + a6 * imgOut[i - 1, :] + b1 * y1[i - 1, :] + b2 * y1[i - 2, :]
//!     y2[-1, :] = 0.0
//!     y2[-2, :] = a7 * imgOut[-1, :]
//!     for i in range(imgIn.shape[0] - 3, -1, -1):
//!         y2[i, :] = a7 * imgOut[i + 1, :] + a8 * imgOut[i + 2, :] + b1 * y2[i + 1, :] + b2 * y2[i + 2, :]
//!     imgOut[:] = c2 * (y1 + y2)
//!     return imgOut
//! ```
//!
//! The row pass (a1..a4) and the column pass (a5..a8) use the same
//! coefficients and the same recurrences, so one function runs both: the
//! column pass is the row pass on the transposed view.
use crate::prelude::*;

/// The filter's coefficients: `a1..a4` (equal to `a5..a8`), `b1`, `b2`.
struct Coefficients {
    a: [Q32; 4],
    b1: Q32,
    b2: Q32,
}

impl Coefficients {
    fn new(alpha: Q32) -> Coefficients {
        let one = Q32::ONE;
        let two = Q32::from_int(2);
        let k = (one - (-alpha).exp()) * (one - (-alpha).exp())
            / (one + alpha * (-alpha).exp() - (two * alpha).exp());
        Coefficients {
            a: [
                k,
                k * (-alpha).exp() * (alpha - one),
                k * (-alpha).exp() * (alpha + one),
                -k * (-two * alpha).exp(),
            ],
            b1: two.powf(-alpha),
            b2: -(-two * alpha).exp(),
        }
    }

    /// One pass along the second axis of `input` (a row pass; a column pass
    /// on transposed views): the causal recurrence `y1` plus the anti-causal
    /// `y2`, the sum written to `out` (`c1 = c2 = 1`).
    fn pass(&self, input: ArrayView2<Q32>, mut out: ArrayViewMut2<Q32>) {
        let [a1, a2, a3, a4] = self.a;
        let (b1, b2) = (self.b1, self.b2);
        let len = input.ncols();
        assert!(len >= 2, "index -2 is out of bounds");
        let mut y1 = vec![Q32::ZERO; len];
        let mut y2 = vec![Q32::ZERO; len];
        for (x, mut out) in input.rows().into_iter().zip(out.rows_mut()) {
            y1[0] = a1 * x[0];
            y1[1] = a1 * x[1] + a2 * x[0] + b1 * y1[0];
            for j in 2..len {
                y1[j] = a1 * x[j] + a2 * x[j - 1] + b1 * y1[j - 1] + b2 * y1[j - 2];
            }
            y2[len - 1] = Q32::ZERO;
            y2[len - 2] = a3 * x[len - 1];
            for j in (0..len - 2).rev() {
                y2[j] = a3 * x[j + 1] + a4 * x[j + 2] + b1 * y2[j + 1] + b2 * y2[j + 2];
            }
            for j in 0..len {
                out[j] = y1[j] + y2[j];
            }
        }
    }
}

pub mod kernel {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let img_in: Array2<Q32> = args.array2();
        let coefficients = Coefficients::new(alpha);
        let mut rows_filtered = Array2::zeros(img_in.dim());
        coefficients.pass(img_in.view(), rows_filtered.view_mut());
        let mut img_out = Array2::zeros(img_in.dim());
        coefficients.pass(rows_filtered.t(), img_out.view_mut().reversed_axes());
        img_out
    }
}
