//! NPBench polybench/jacobi_2d: `TSTEPS - 1` rounds of a five-point Jacobi
//! stencil, ping-ponging between `A` and `B`, in place.
//!
//! ```python
//! def kernel(TSTEPS: Integer, A: Float[Array, '150 150'], B: Float[Array, '150 150']):
//!     for t in range(1, TSTEPS):
//!         B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1])
//!         A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1])
//! ```
use crate::prelude::*;

/// `round(0.2 * 2^32)`.
const FIFTH: Q32 = Q32::from_bits(858_993_459);

/// `dst[1:-1, 1:-1] = 0.2 * (centre + west + east + south + north)` of `src`.
fn relax(src: &Array2<Q32>, dst: &mut Array2<Q32>) {
    assert_eq!(src.dim(), dst.dim(), "A and B have different shapes");
    let (rows, cols) = src.dim();
    for i in 1..rows.saturating_sub(1) {
        for j in 1..cols.saturating_sub(1) {
            let sum =
                src[(i, j)] + src[(i, j - 1)] + src[(i, j + 1)] + src[(i + 1, j)] + src[(i - 1, j)];
            dst[(i, j)] = FIFTH * sum;
        }
    }
}

pub mod kernel {
    use super::*;

    // outputs: mutated A, B
    pub type Output = (Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tsteps: i64 = args.scalar();
        let mut a: Array2<Q32> = args.array2();
        let mut b: Array2<Q32> = args.array2();
        for _ in 1..tsteps {
            relax(&a, &mut b);
            relax(&b, &mut a);
        }
        (a, b)
    }
}
