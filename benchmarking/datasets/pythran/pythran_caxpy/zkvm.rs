//! Pythran caxpy: BLAS `CAXPY`, `CY := CY + CA * CX` over strided complex
//! vectors.
//!
//! ```python
//! def CAXPY(N: Integer, CA: Complex, CX: Complex[Array, '4'], INCX: Integer, CY: Complex[Array, '4'], INCY: Integer):
//!     if N <= 0:
//!         return
//!     if abs(CA) == 0.0:
//!         return
//!     if INCX == 1 and INCY == 1:
//!         for I in range(N):
//!             CY[I] = CY[I] + CA * CX[I]
//!     else:
//!         IX = 0
//!         IY = 0
//!         if INCX < 0:
//!             IX = (-N + 1) * INCX
//!         if INCY < 0:
//!             IY = (-N + 1) * INCY
//!         for I in range(N):
//!             CY[IY] = CY[IY] + CA * CX[IX]
//!             IX = IX + INCX
//!             IY = IY + INCY
//!     return CY
//! ```
use crate::prelude::*;

#[allow(non_snake_case)]
pub mod CAXPY {
    use super::*;

    /// `None` where the kernel returns early (Python `return` without a value).
    pub type Output = Option<Array1<C32>>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let ca: C32 = args.scalar();
        let cx: Array1<C32> = args.array1();
        let incx: i64 = args.scalar();
        let mut cy: Array1<C32> = args.array1();
        let incy: i64 = args.scalar();
        if n <= 0 || ca == C32::ZERO {
            return None;
        }
        // The unit-stride branch is the strided loop with both starts at 0.
        let start = |inc: i64| if inc < 0 { (1 - n) * inc } else { 0 };
        let (mut ix, mut iy) = (start(incx), start(incy));
        for _ in 0..n {
            let (x, y) = (np_index(ix, cx.len()), np_index(iy, cy.len()));
            cy[y] = cy[y] + ca * cx[x];
            ix += incx;
            iy += incy;
        }
        Some(cy)
    }
}
