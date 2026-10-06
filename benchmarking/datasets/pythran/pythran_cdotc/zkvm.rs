//! Pythran cdotc: BLAS `CDOTC`, the dot product `sum(conj(CX) * CY)` over
//! strided complex vectors.
//!
//! ```python
//! @requires(lambda N, CX, INCX, CY, INCY: 0 <= N <= 1024)
//! def CDOTC(N: Integer, CX: Complex[Array, '4'], INCX: Integer, CY: Complex[Array, '4'], INCY: Integer):
//!     CTEMP = complex(0.0, 0.0)
//!     CDOTC = complex(0.0, 0.0)
//!     if N <= 0:
//!         return
//!     if INCX == 1 and INCY == 1:
//!         for I in range(N):
//!             CTEMP = CTEMP + CX[I].conjugate() * CY[I]
//!     else:
//!         IX = 0
//!         IY = 0
//!         if INCX < 0:
//!             IX = (-N + 1) * INCX
//!         if INCY < 0:
//!             IY = (-N + 1) * INCY
//!         for I in range(N):
//!             CTEMP = CTEMP + CX[IX].conjugate() * CY[IY]
//!             IX = IX + INCX
//!             IY = IY + INCY
//!     return CTEMP
//! ```
use crate::prelude::*;

#[allow(non_snake_case)]
pub mod CDOTC {
    use super::*;

    /// `None` where the kernel returns early (Python `return` without a value).
    pub type Output = Option<C32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n: i64 = args.scalar();
        let cx: Array1<C32> = args.array1();
        let incx: i64 = args.scalar();
        let cy: Array1<C32> = args.array1();
        let incy: i64 = args.scalar();
        assert!((0..=1024).contains(&n), "requires 0 <= N <= 1024");
        if n <= 0 {
            return None;
        }
        // The unit-stride branch is the strided loop with both starts at 0.
        let start = |inc: i64| if inc < 0 { (1 - n) * inc } else { 0 };
        let (mut ix, mut iy) = (start(incx), start(incy));
        let mut ctemp = C32::ZERO;
        for _ in 0..n {
            ctemp += cx[np_index(ix, cx.len())].conj() * cy[np_index(iy, cy.len())];
            ix += incx;
            iy += incy;
        }
        Some(ctemp)
    }
}
