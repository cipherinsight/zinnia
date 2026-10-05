//! NPBench polybench/heat_3d: `TSTEPS - 1` double steps of the explicit 3-D
//! heat equation, alternating between `A` and `B`, in place.
//!
//! ```python
//! def kernel(TSTEPS: Integer, A: Float[Array, '25 25 25'], B: Float[Array, '25 25 25']):
//!     for t in range(1, TSTEPS):
//!         B[1:-1, 1:-1, 1:-1] = 0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1]) + 0.125 * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1]) + 0.125 * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1]
//!         A[1:-1, 1:-1, 1:-1] = 0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1]) + 0.125 * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1]) + 0.125 * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1]
//! ```
use crate::prelude::*;

const EIGHTH: Q32 = Q32::from_bits(1 << 29); // 0.125

/// `dst[1:-1, 1:-1, 1:-1] = ` the heat stencil of `src`.
fn step(src: &Array3<Q32>, dst: &mut Array3<Q32>) {
    let (ni, nj, nk) = src.dim();
    assert_eq!(dst.dim(), src.dim(), "A and B differ in shape");
    for i in 1..ni.saturating_sub(1) {
        for j in 1..nj.saturating_sub(1) {
            for k in 1..nk.saturating_sub(1) {
                let c = src[(i, j, k)];
                dst[(i, j, k)] = EIGHTH * (src[(i + 1, j, k)] - c * 2 + src[(i - 1, j, k)])
                    + EIGHTH * (src[(i, j + 1, k)] - c * 2 + src[(i, j - 1, k)])
                    + EIGHTH * (src[(i, j, k + 1)] - c * 2 + src[(i, j, k - 1)])
                    + c;
            }
        }
    }
}

pub mod kernel {
    use super::*;

    // outputs: mutated A, B
    pub type Output = (Array3<Q32>, Array3<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tsteps: i64 = args.scalar();
        let mut a: Array3<Q32> = args.array3();
        let mut b: Array3<Q32> = args.array3();
        for _ in 1..tsteps {
            step(&a, &mut b);
            step(&b, &mut a);
        }
        (a, b)
    }
}
