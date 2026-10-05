//! NPBench deep_learning/resnet: a ResNet basic block (three valid 2-D
//! convolutions in NHWC layout, batch normalisation over the batch axis,
//! ReLU, and the residual connection).
//!
//! ```python
//! def relu(x):
//!     return np.maximum(x, 0)
//!
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
//! def batchnorm2d(x, eps=1e-05):
//!     mean = np.mean(x, axis=0, keepdims=True)
//!     std = np.std(x, axis=0, keepdims=True)
//!     return (x - mean) / np.sqrt(std + eps)
//!
//! def resnet_basicblock(input: Float[Array, '8 14 14 32'], conv1: Float[Array, '1 1 32 8'], conv2: Float[Array, '3 3 8 8'], conv3: Float[Array, '1 1 8 32']):
//!     padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))
//!     padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
//!     x = batchnorm2d(padded)
//!     x = relu(x)
//!     x = conv2d(x, conv2)
//!     x = batchnorm2d(x)
//!     x = relu(x)
//!     x = conv2d(x, conv3)
//!     x = batchnorm2d(x)
//!     return relu(x + input)
//! ```
//!
//! `conv2d` stores its result as float32 in NumPy; here every value is Q32.
use crate::prelude::*;

/// `eps = 1e-05`: `round(1e-5 * 2^32)`.
const EPS: Q32 = Q32::from_bits(42_950);

/// `np.maximum(x, 0)`.
fn relu(x: Array4<Q32>) -> Array4<Q32> {
    x.mapv_into(|v| v.max(Q32::ZERO))
}

/// Valid 2-D convolution of NHWC `input` with `K x K x C_in x C_out`
/// `weights`: the sum of the elementwise products over each window (each
/// product rescaled, as `np.sum` of a product array).
fn conv2d(input: &Array4<Q32>, weights: &Array4<Q32>) -> Array4<Q32> {
    let (k, kw, c_in, c_out) = weights.dim();
    let (n, h, w, c) = input.dim();
    assert!(
        kw == k && c_in == c,
        "conv2d: the weights do not match the input"
    );
    let h_out = (h + 1)
        .checked_sub(k)
        .expect("negative dimensions are not allowed");
    let w_out = (w + 1)
        .checked_sub(k)
        .expect("negative dimensions are not allowed");
    Array4::from_shape_fn((n, h_out, w_out, c_out), |(b, i, j, co)| {
        let mut acc = Q32::ZERO;
        for di in 0..k {
            for dj in 0..k {
                for ci in 0..c {
                    acc += input[(b, i + di, j + dj, ci)] * weights[(di, dj, ci, co)];
                }
            }
        }
        acc
    })
}

/// `(x - mean) / np.sqrt(std + eps)`, with the mean and the population
/// standard deviation over the batch axis.
fn batchnorm2d(mut x: Array4<Q32>) -> Array4<Q32> {
    let n = x.len_of(Axis(0)) as i64;
    for mut lane in x.lanes_mut(Axis(0)) {
        let mean = lane.sum() / n;
        let variance = lane.iter().map(|&v| (v - mean) * (v - mean)).sum::<Q32>() / n;
        let denominator = (variance.sqrt() + EPS).sqrt();
        lane.mapv_inplace(|v| (v - mean) / denominator);
    }
    x
}

pub mod resnet_basicblock {
    use super::*;

    pub type Output = Array4<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input: Array4<Q32> = args.array4();
        let conv1: Array4<Q32> = args.array4();
        let conv2: Array4<Q32> = args.array4();
        let conv3: Array4<Q32> = args.array4();
        let (n, h, w, _) = input.dim();
        let mut padded = Array4::<Q32>::zeros((n, h + 2, w + 2, conv1.len_of(Axis(3))));
        // `assign` broadcasts like NumPy's slice assignment.
        padded
            .slice_mut(s![.., 1..h + 1, 1..w + 1, ..])
            .assign(&conv2d(&input, &conv1));
        let x = relu(batchnorm2d(padded));
        let x = relu(batchnorm2d(conv2d(&x, &conv2)));
        let x = batchnorm2d(conv2d(&x, &conv3));
        relu(&x + &input)
    }
}
