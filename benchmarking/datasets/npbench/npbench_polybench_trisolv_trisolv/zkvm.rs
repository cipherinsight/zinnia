//! NPBench polybench/trisolv: forward substitution `L x = b`, into `x`.
//!
//! ```python
//! def kernel(L: Float[Array, '2000 2000'], x: Float[Array, '2000'], b: Float[Array, '2000']):
//!     for i in range(x.shape[0]):
//!         x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated x
    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let l: Array2<Q32> = args.array2();
        let mut x: Array1<Q32> = args.array1();
        let b: Array1<Q32> = args.array1();
        for i in 0..x.len() {
            let solved = np::inner(l.slice(s![i, ..i]), x.slice(s![..i]));
            x[i] = (b[i] - solved) / l[(i, i)];
        }
        x
    }
}
