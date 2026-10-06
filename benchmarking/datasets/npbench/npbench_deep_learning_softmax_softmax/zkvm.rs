//! NPBench deep_learning/softmax: softmax over the last axis of a 4-D array.
//!
//! ```python
//! def softmax(x: Float[Array, '16 16 128 128']):
//!     tmp_max = np.max(x, axis=-1, keepdims=True)
//!     tmp_out = np.exp(x - tmp_max)
//!     tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
//!     return tmp_out / tmp_sum
//! ```
use crate::prelude::*;

pub mod softmax {
    use super::*;

    pub type Output = Array4<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let mut x: Array4<Q32> = args.array4();
        for mut lane in x.lanes_mut(Axis(3)) {
            let max = *lane
                .iter()
                .max()
                .expect("zero-size array to reduction operation maximum");
            lane.mapv_inplace(|v| (v - max).exp());
            let sum: Q32 = lane.sum();
            lane.mapv_inplace(|v| v / sum);
        }
        x
    }
}
