//! NPBench polybench/doitgen: multiplies every row `A[r, q, :]` by `C4`, in
//! place.
//!
//! ```python
//! def kernel(NR: Integer, NQ: Integer, NP: Integer, A: Float[Array, '60 60 128'], C4: Float[Array, '128 128']):
//!     A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A
    pub type Output = Array3<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let n_r: i64 = args.scalar();
        let n_q: i64 = args.scalar();
        let n_p: i64 = args.scalar();
        let mut a: Array3<Q32> = args.array3();
        let c4: Array2<Q32> = args.array2();
        // The reshapes keep A's element count and the assignment its shape,
        // so A is (NR, NQ, NP); `@ C4` then needs C4 to be NP x NP.
        let dim = |d: i64| usize::try_from(d).expect("negative dimensions are not allowed");
        assert_eq!(a.dim(), (dim(n_r), dim(n_q), dim(n_p)), "cannot reshape A into (NR, NQ, NP)");
        assert_eq!(c4.dim(), (dim(n_p), dim(n_p)), "shapes not aligned");
        for mut row in a.lanes_mut(Axis(2)) {
            let product = np::vecmat(row.view(), c4.view());
            row.assign(&product);
        }
        a
    }
}
