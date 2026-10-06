//! NPBench polybench/gemver: a rank-2 update of `A` followed by two
//! matrix-vector products, in place.
//!
//! ```python
//! def kernel(alpha: Float, beta: Float, A: Float[Array, '1000 1000'], u1: Float[Array, '1000'], v1: Float[Array, '1000'], u2: Float[Array, '1000'], v2: Float[Array, '1000'], w: Float[Array, '1000'], x: Float[Array, '1000'], y: Float[Array, '1000'], z: Float[Array, '1000']):
//!     A += np.outer(u1, v1) + np.outer(u2, v2)
//!     x += beta * y @ A + z
//!     w += alpha * A @ x
//! ```
use crate::prelude::*;

pub mod kernel {
    use super::*;

    // outputs: mutated A, w, x
    pub type Output = (Array2<Q32>, Array1<Q32>, Array1<Q32>);

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let alpha: Q32 = args.scalar();
        let beta: Q32 = args.scalar();
        let mut a: Array2<Q32> = args.array2();
        let u1: Array1<Q32> = args.array1();
        let v1: Array1<Q32> = args.array1();
        let u2: Array1<Q32> = args.array1();
        let v2: Array1<Q32> = args.array1();
        let mut w: Array1<Q32> = args.array1();
        let mut x: Array1<Q32> = args.array1();
        let y: Array1<Q32> = args.array1();
        let z: Array1<Q32> = args.array1();
        assert!(
            (u1.len(), v1.len()) == a.dim() && (u2.len(), v2.len()) == a.dim(),
            "operands could not be broadcast together"
        );
        // A += np.outer(u1, v1) + np.outer(u2, v2)
        Zip::indexed(&mut a).for_each(|(i, j), a| *a += u1[i] * v1[j] + u2[i] * v2[j]);
        // x += (beta * y) @ A + z: `*` and `@` group left to right.
        let ya = np::vecmat(y.mapv(|v| beta * v).view(), a.view());
        Zip::from(&mut x)
            .and(&ya)
            .and(&z)
            .for_each(|x, &p, &z| *x += p + z);
        // w += (alpha * A) @ x, with the updated x.
        w += &np::matvec(a.mapv(|v| alpha * v).view(), x.view());
        (a, w, x)
    }
}
