//! Pythran lap: the fourth-order periodic 1-D Laplacian of `var`, divided by
//! `dh2`.
//!
//! ```python
//! def laplacian(var: Float[Array, '8'], dh2: Float):
//!     lap = numpy.zeros_like(var)
//!     lap[1:] = 4.0 / 3.0 * var[:-1]
//!     lap[0] = 4.0 / 3.0 * var[1]
//!     lap[:-1] += 4.0 / 3.0 * var[1:]
//!     lap[-1] += 4.0 / 3.0 * var[0]
//!     lap += -5.0 / 2.0 * var
//!     lap[2:] += -1.0 / 12.0 * var[:-2]
//!     lap[:2] += -1.0 / 12.0 * var[-2:]
//!     lap[:-2] += -1.0 / 12.0 * var[2:]
//!     lap[-2:] += -1.0 / 12.0 * var[:2]
//!     return lap / dh2
//! ```
use crate::prelude::*;

/// Python's `4.0 / 3.0`, `-5.0 / 2.0`, `-1.0 / 12.0`, quantized.
const NEAR: Q32 = Q32::from_bits(5_726_623_061);
const CENTER: Q32 = Q32::from_bits(-10_737_418_240);
const FAR: Q32 = Q32::from_bits(-357_913_941);

/// `lap[dst] += c * var[src]` for two equally long slices.
fn add_scaled(mut lap: ArrayViewMut1<Q32>, c: Q32, var: ArrayView1<Q32>) {
    assert_eq!(lap.len(), var.len(), "operands could not be broadcast together");
    lap.zip_mut_with(&var, |l, &v| *l += c * v);
}

pub mod laplacian {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let var: Array1<Q32> = args.array1();
        let dh2: Q32 = args.scalar();
        assert!(dh2 != Q32::ZERO, "requires dh2 != 0.0");
        let mut lap = Array1::<Q32>::zeros(var.len());
        lap.slice_mut(s![1..]).zip_mut_with(&var.slice(s![..-1]), |l, &v| *l = NEAR * v);
        lap[0] = NEAR * var[1];
        add_scaled(lap.slice_mut(s![..-1]), NEAR, var.slice(s![1..]));
        lap[np_index(-1, var.len())] += NEAR * var[0];
        add_scaled(lap.view_mut(), CENTER, var.view());
        add_scaled(lap.slice_mut(s![2..]), FAR, var.slice(s![..-2]));
        add_scaled(lap.slice_mut(s![..2]), FAR, var.slice(s![-2..]));
        add_scaled(lap.slice_mut(s![..-2]), FAR, var.slice(s![2..]));
        add_scaled(lap.slice_mut(s![-2..]), FAR, var.slice(s![..2]));
        lap.mapv(|l| l / dh2)
    }
}
