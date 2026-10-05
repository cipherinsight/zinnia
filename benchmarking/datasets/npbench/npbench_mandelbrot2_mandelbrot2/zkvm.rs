//! NPBench mandelbrot2: escape times of the Mandelbrot iteration on an
//! `xn x yn` grid, iterating only the points that have not escaped yet;
//! each escaped point records its iteration count and its value at escape.
//!
//! ```python
//! @requires(lambda xmin, xmax, ymin, ymax, xn, yn, itermax, horizon: 0 <= itermax <= 1024)
//! def mandelbrot(xmin: Float, xmax: Float, ymin: Float, ymax: Float, xn: Integer, yn: Integer, itermax: Integer, horizon: Float=2.0):
//!     Xi, Yi = np.mgrid[0:xn, 0:yn]
//!     X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
//!     Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
//!     C = X + Y * 1j
//!     N_ = np.zeros(C.shape, dtype=np.int64)
//!     Z_ = np.zeros(C.shape, dtype=np.complex128)
//!     Xi.shape = Yi.shape = C.shape = xn * yn
//!     Z = np.zeros(C.shape, np.complex128)
//!     for i in range(itermax):
//!         if not len(Z):
//!             break
//!         np.multiply(Z, Z, Z)
//!         np.add(Z, C, Z)
//!         I = abs(Z) > horizon
//!         N_[Xi[I], Yi[I]] = i + 1
//!         Z_[Xi[I], Yi[I]] = Z[I]
//!         np.logical_not(I, I)
//!         Z = Z[I]
//!         Xi, Yi = (Xi[I], Yi[I])
//!         C = C[I]
//!     return (Z_.T, N_.T)
//! ```
//!
//! The points evolve independently, so each is iterated on its own until it
//! escapes (the kernel's compaction of the live points is bookkeeping).
use crate::prelude::*;

/// `np.linspace(start, stop, num)`: `start + k * step` with
/// `step = (stop - start) / (num - 1)`, and the last point exactly `stop`.
fn linspace(start: Q32, stop: Q32, num: i64) -> Array1<Q32> {
    let len = usize::try_from(num).expect("Number of samples must be non-negative");
    let (div, delta) = (num - 1, stop - start);
    let mut y: Array1<Q32> = if div <= 0 {
        Array1::from_elem(len, start)
    } else {
        let step = delta / div;
        if step == Q32::ZERO {
            // NumPy's path for a step that rounds to zero: `k / div * delta`.
            (0..num)
                .map(|k| Q32::from_int(k) / div * delta + start)
                .collect()
        } else {
            (0..num).map(|k| step * k + start).collect()
        }
    };
    if len > 1 {
        y[len - 1] = stop;
    }
    y
}

pub mod mandelbrot {
    use super::*;

    /// `(Z_.T, N_.T)`, both `yn x xn`.
    pub type Output = (Array2<C32>, Array2<i64>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let [xmin, xmax, ymin, ymax]: [Q32; 4] = core::array::from_fn(|_| args.scalar());
        let xn: i64 = args.scalar();
        let yn: i64 = args.scalar();
        let itermax: i64 = args.scalar();
        let horizon: Q32 = args.scalar();
        assert!(
            (0..=1024).contains(&itermax),
            "requires 0 <= itermax <= 1024"
        );
        let xs = linspace(xmin, xmax, xn);
        let ys = linspace(ymin, ymax, yn);
        let mut escaped_z = Array2::<C32>::zeros((ys.len(), xs.len()));
        let mut escaped_at = Array2::<i64>::zeros((ys.len(), xs.len()));
        for (xi, &x) in xs.iter().enumerate() {
            for (yi, &y) in ys.iter().enumerate() {
                let c = C32::new(x, y);
                let mut z = C32::ZERO;
                for i in 0..itermax {
                    z = z * z + c;
                    if z.abs() > horizon {
                        escaped_at[(yi, xi)] = i + 1;
                        escaped_z[(yi, xi)] = z;
                        break;
                    }
                }
            }
        }
        (escaped_z, escaped_at)
    }
}
