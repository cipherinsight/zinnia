//! Pythran projection_simplex: Euclidean projection of `v` onto the simplex
//! `{w >= 0, sum(w) = z}` by sorting.
//!
//! ```python
//! def projection_simplex(v: Float[Array, '10'], z: Integer=1):
//!     n_features = v.shape[0]
//!     u = np.sort(v)[::-1]
//!     cssv = np.cumsum(u) - z
//!     ind = np.arange(n_features) + 1
//!     cond = u - cssv / ind > 0
//!     rho = ind[cond][-1]
//!     theta = cssv[cond][-1] / float(rho)
//!     w = np.maximum(v - theta, 0)
//!     return w
//! ```
use crate::prelude::*;

pub mod projection_simplex {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let v: Array1<Q32> = args.array1();
        let z: i64 = args.scalar();
        let mut u = v.to_vec();
        u.sort_unstable_by(|a, b| b.cmp(a));
        // The last `ind` where `cond` holds, with its `cssv`.
        let mut last = None;
        let mut cumsum = Q32::ZERO;
        for (k, &uk) in u.iter().enumerate() {
            cumsum += uk;
            let cssv = cumsum - Q32::from_int(z);
            let ind = k as i64 + 1;
            if uk - cssv / ind > Q32::ZERO {
                last = Some((ind, cssv));
            }
        }
        let (rho, cssv) = last.expect("index -1 is out of bounds: cond is all false");
        let theta = cssv / rho;
        v.mapv(|vi| (vi - theta).max(Q32::ZERO))
    }
}
