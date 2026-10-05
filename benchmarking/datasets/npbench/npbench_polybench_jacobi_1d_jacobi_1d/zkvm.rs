//! NPBench polybench/jacobi_1d: a three-point averaging stencil applied
//! alternately from `A` into `B` and back, in place.
//!
//! ```python
//! def kernel(TSTEPS: Integer, A: Float[Array, '3200'], B: Float[Array, '3200']):
//!     for t in range(1, TSTEPS):
//!         B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
//!         A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
//! ```
use crate::prelude::*;

/// The kernel's `0.33333`, quantized: `round(0.33333 * 2^32)`.
const WEIGHT: Q32 = Q32::from_bits(1_431_641_449);

/// `dst[1:-1] = 0.33333 * (src[:-2] + src[1:-1] + src[2:])`.
fn smooth(dst: &mut Array1<Q32>, src: &Array1<Q32>) {
    for i in 1..src.len().saturating_sub(1) {
        dst[i] = WEIGHT * (src[i - 1] + src[i] + src[i + 1]);
    }
}

pub mod kernel {
    use super::*;

    // outputs: mutated A, B
    pub type Output = (Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let tsteps: i64 = args.scalar();
        let mut a: Array1<Q32> = args.array1();
        let mut b: Array1<Q32> = args.array1();
        assert_eq!(a.len(), b.len(), "operands could not be broadcast together");
        for _ in 1..tsteps {
            smooth(&mut b, &a);
            smooth(&mut a, &b);
        }
        (a, b)
    }
}
