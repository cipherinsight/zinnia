//! Pythran ccopy: BLAS `CCOPY`, `CY := CX` over strided complex vectors.
//!
//! ```python
//! @requires(lambda N, CX, INCX, CY, INCY: 0 <= N <= 1024)
//! def CCOPY(N: Integer, CX: Complex[Array, '4'], INCX: Integer, CY: Complex[Array, '4'], INCY: Integer):
//!     if N <= 0:
//!         return
//!     if INCX == 1 and INCY == 1:
//!         for I in range(N):
//!             CY[I] = CX[I]
//!     else:
//!         IX = 0
//!         IY = 0
//!         if INCX < 0:
//!             IX = (-N + 1) * INCX
//!         if INCY < 0:
//!             IY = (-N + 1) * INCY
//!         for I in range(N):
//!             CY[IY] = CX[IX]
//!             IX = IX + INCX
//!             IY = IY + INCY
//!     return
//! ```
use crate::prelude::*;

#[allow(non_snake_case)]
pub mod CCOPY {
    use super::*;

    // outputs: mutated CY
    pub type Output = Array1<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let cx: Array1<C32> = args.array1();
        let incx: i64 = args.scalar();
        let mut cy: Array1<C32> = args.array1();
        let incy: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        // The unit-stride branch is the strided loop with both starts at 0, and
        // the loop does nothing for N <= 0 (the early return).
        let start = |inc: i64| if inc < 0 { (1 - n) * inc } else { 0 };
        let (mut ix, mut iy) = (start(incx), start(incy));
        for _ in 0..n {
            let y = np_index(iy, cy.len());
            cy[y] = cx[np_index(ix, cx.len())];
            ix += incx;
            iy += incy;
        }
        cy
    }
}
