//! NPBench compute: `np.clip(array_1, 2, 10) * a + array_2 * b + c` on
//! integer matrices.
//!
//! ```python
//! def compute(array_1: Integer[Array, '2000 2000'], array_2: Integer[Array, '2000 2000'], a: Integer, b: Integer, c: Integer):
//!     return np.clip(array_1, 2, 10) * a + array_2 * b + c
//! ```
use crate::prelude::*;

pub mod compute {
    use super::*;

    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let array_1: Array2<i64> = args.array2();
        let array_2: Array2<i64> = args.array2();
        let a: i64 = args.scalar();
        let b: i64 = args.scalar();
        let c: i64 = args.scalar();
        // int64 arithmetic wraps.
        Zip::from(&array_1).and(&array_2).map_collect(|&x, &y| {
            x.clamp(2, 10)
                .wrapping_mul(a)
                .wrapping_add(y.wrapping_mul(b))
                .wrapping_add(c)
        })
    }
}
