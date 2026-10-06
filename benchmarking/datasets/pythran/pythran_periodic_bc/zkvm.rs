//! Pythran periodic_bc: periodic boundary condition on the second axis,
//! copying the second-to-last plane onto the first; returns `f`.
//!
//! ```python
//! def periodic_bc(f: Float[Array, '3 3 3']):
//!     f[:, 0, :] = f[:, -2, :]
//!     return f
//! ```
use crate::prelude::*;

pub mod periodic_bc {
    use super::*;

    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut f: Array3<Q32> = args.array3();
        let (n0, n1, n2) = f.dim();
        // `-2` needs `n1 >= 2`, which also makes index 0 valid.
        let source = np_index(-2, n1);
        for i in 0..n0 {
            for k in 0..n2 {
                f[(i, 0, k)] = f[(i, source, k)];
            }
        }
        f
    }
}
