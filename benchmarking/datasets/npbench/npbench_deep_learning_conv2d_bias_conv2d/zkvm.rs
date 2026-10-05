//! NPBench deep_learning/conv2d_bias: a valid 2-D convolution (NHWC input,
//! `K x K x C_in x C_out` weights) plus a bias per output channel.
//!
//! ```python
//! def conv2d(input, weights):
//!     K = weights.shape[0]
//!     N = input.shape[0]
//!     H_out = input.shape[1] - K + 1
//!     W_out = input.shape[2] - K + 1
//!     C_out = weights.shape[3]
//!     output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
//!     for i in range(H_out):
//!         for j in range(W_out):
//!             output[:, i, j, :] = np.sum(input[:, i:i + K, j:j + K, :, np.newaxis] * weights[np.newaxis, :, :, :], axis=(1, 2, 3))
//!     return output
//!
//! def conv2d_bias(input: Float[Array, '8 32 32 3'], weights: Float[Array, '2 2 3 16'], bias: Float[Array, '16']):
//!     return conv2d(input, weights) + bias
//! ```
//!
//! The window is an elementwise product followed by `np.sum` (not a
//! contraction), so every product is rescaled on its own. NumPy stores the
//! convolution as float32; here it is Q32 like every float.
use crate::prelude::*;

pub mod conv2d_bias {
    use super::*;

    pub type Output = Array4<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input: Array4<Q32> = args.array4();
        let weights: Array4<Q32> = args.array4();
        let bias: Array1<Q32> = args.array1();
        let (n, height, width, c_in) = input.dim();
        let (k, k2, w_in, c_out) = weights.dim();
        assert!(k2 == k && w_in == c_in, "operands could not be broadcast together");
        let out_extent = |extent: usize| (extent + 1).checked_sub(k).expect("negative dimensions are not allowed");
        let (h_out, w_out) = (out_extent(height), out_extent(width));
        let mut output = Array4::zeros((n, h_out, w_out, c_out));
        for ((b, i, j, co), out) in output.indexed_iter_mut() {
            let window = input.slice(s![b, i..i + k, j..j + k, ..]);
            let filter = weights.slice(s![.., .., .., co]);
            *out = Zip::from(window).and(filter).fold(Q32::ZERO, |sum, &x, &w| sum + x * w);
        }
        output += &bias;
        output
    }
}
