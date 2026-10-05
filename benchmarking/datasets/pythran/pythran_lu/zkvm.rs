//! Pythran lu: LU factorization with partial pivoting, in place, written two
//! ways (NumPy row swap and `argmax`, or explicit loops). Both compute the
//! same thing, so both entry functions run the same elimination.
//!
//! ```python
//! def factorMatrix0(M: Float[Array, '5 5']):
//!     n = M.shape[0]
//!     m = M.shape[1]
//!     for line in range(0, n - 1):
//!         cmax = np.argmax(abs(M[line:n, line])) + line
//!         if cmax != line:
//!             M[[line, cmax]] = M[[cmax, line]]
//!         pivot = M[line, line]
//!         for j in range(line + 1, n):
//!             v = M[j, line] / pivot
//!             for k in range(line, m):
//!                 M[j, k] -= v * M[line, k]
//!
//! def factorMatrix1(M: Float[Array, '5 5']):
//!     (the same with the pivot search and the row swap as loops: the first
//!     row with the largest |M[i, line]| wins, as with np.argmax)
//! ```
use crate::prelude::*;

/// Gaussian elimination with partial pivoting on `m`, in place.
fn factor(m: &mut Array2<Q32>) {
    let (n, cols) = m.dim();
    for line in 0..n.saturating_sub(1) {
        let magnitudes: Vec<Q32> = m.slice(s![line.., line]).iter().map(|v| v.abs()).collect();
        let pivot_row = line + np::argmax(&magnitudes);
        if pivot_row != line {
            for k in 0..cols {
                m.swap((line, k), (pivot_row, k));
            }
        }
        let pivot = m[(line, line)];
        for j in line + 1..n {
            let v = m[(j, line)] / pivot;
            for k in line..cols {
                let update = v * m[(line, k)];
                m[(j, k)] -= update;
            }
        }
    }
}

#[allow(non_snake_case)]
pub mod factorMatrix0 {
    use super::*;

    // outputs: mutated M
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut m: Array2<Q32> = args.array2();
        factor(&mut m);
        m
    }
}

#[allow(non_snake_case)]
pub mod factorMatrix1 {
    use super::*;

    // outputs: mutated M
    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut m: Array2<Q32> = args.array2();
        factor(&mut m);
        m
    }
}
