//! Pythran lapl2d: `niter` Jacobi sweeps of the 5-point Laplacian, ping-ponging
//! between the two grids.
//!
//! ```python
//! @requires(lambda In, Out, niter: 0 <= niter <= 1024)
//! def lapl2d(In: Float[Array, '4 4'], Out: Float[Array, '4 4'], niter: Integer):
//!     siz = In.shape[0]
//!     h2 = (1.0 / siz) ** 2
//!     for it in range(0, niter):
//!         Out[1:siz - 1, 1:siz - 1] = h2 * (In[0:siz - 2, 1:siz - 1] + In[1:siz - 1, 0:siz - 2] - 4.0 * In[1:siz - 1, 1:siz - 1] + In[2:siz, 1:siz - 1] + In[1:siz - 1, 2:siz])
//!         In, Out = (Out, In)
//! ```
use crate::prelude::*;

pub mod lapl2d {
    use super::*;

    // outputs: mutated In, Out
    pub type Output = (Array2<Q32>, Array2<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut grid_in: Array2<Q32> = args.array2();
        let mut grid_out: Array2<Q32> = args.array2();
        let niter: i64 = args.scalar();
        assert!((0..=1024).contains(&niter), "requires 0 <= niter <= 1024");
        let siz = grid_in.nrows();
        let h2 = (Q32::ONE / siz as i64).powi(2);
        // `In, Out = (Out, In)` rebinds the names: the sweeps alternate between
        // writing the caller's Out and the caller's In.
        let (mut src, mut dst) = (&mut grid_in, &mut grid_out);
        for _ in 0..niter {
            Zip::from(dst.slice_mut(s![1..siz - 1, 1..siz - 1]))
                .and(src.slice(s![0..siz - 2, 1..siz - 1]))
                .and(src.slice(s![1..siz - 1, 0..siz - 2]))
                .and(src.slice(s![1..siz - 1, 1..siz - 1]))
                .and(src.slice(s![2..siz, 1..siz - 1]))
                .and(src.slice(s![1..siz - 1, 2..siz]))
                .for_each(|out, &north, &west, &centre, &south, &east| {
                    *out = h2 * (north + west - centre * 4 + south + east);
                });
            core::mem::swap(&mut src, &mut dst);
        }
        (grid_in, grid_out)
    }
}
