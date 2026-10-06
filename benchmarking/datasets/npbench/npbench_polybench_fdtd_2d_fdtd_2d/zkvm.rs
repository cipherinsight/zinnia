//! NPBench polybench/fdtd_2d: `TMAX` steps of the 2-D finite-difference
//! time-domain method, in place on `ex`, `ey` and `hz`.
//!
//! ```python
//! def kernel(TMAX: Integer, ex: Float[Array, '200 220'], ey: Float[Array, '200 220'], hz: Float[Array, '200 220'], _fict_: Float[Array, '20']):
//!     for t in range(TMAX):
//!         ey[0, :] = _fict_[t]
//!         ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
//!         ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
//!         hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])
//! ```
use crate::prelude::*;

const C_HZ: Q32 = Q32::from_bits(3_006_477_107); // 0.7

pub mod kernel {
    use super::*;

    // outputs: mutated ex, ey, hz
    pub type Output = (Array2<Q32>, Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tmax: i64 = args.scalar();
        let mut ex: Array2<Q32> = args.array2();
        let mut ey: Array2<Q32> = args.array2();
        let mut hz: Array2<Q32> = args.array2();
        let fict: Array1<Q32> = args.array1();
        let (rows, cols) = hz.dim();
        assert!(ex.dim() == hz.dim() && ey.dim() == hz.dim(), "ex, ey and hz differ in shape");
        let steps = usize::try_from(tmax.max(0)).expect("TMAX exceeds the address space");
        // Each statement reads only arrays it does not write (or, for `hz`,
        // the element it updates), so element-wise loops in place follow it.
        for t in 0..steps {
            ey.row_mut(0).fill(fict[t]);
            for i in 1..rows {
                for j in 0..cols {
                    ey[(i, j)] -= Q32::HALF * (hz[(i, j)] - hz[(i - 1, j)]);
                }
            }
            for i in 0..rows {
                for j in 1..cols {
                    ex[(i, j)] -= Q32::HALF * (hz[(i, j)] - hz[(i, j - 1)]);
                }
            }
            for i in 0..rows.saturating_sub(1) {
                for j in 0..cols.saturating_sub(1) {
                    hz[(i, j)] -=
                        C_HZ * (ex[(i, j + 1)] - ex[(i, j)] + ey[(i + 1, j)] - ey[(i, j)]);
                }
            }
        }
        (ex, ey, hz)
    }
}
