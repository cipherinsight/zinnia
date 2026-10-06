//! NPBench deep_learning/lenet: LeNet-5 inference (two convolution + ReLU +
//! 2x2 max-pool stages, then three dense layers) on a batch of NHWC images.
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
//! def maxpool2d(x):
//!     output = np.empty([x.shape[0], x.shape[1] // 2, x.shape[2] // 2, x.shape[3]], dtype=x.dtype)
//!     for i in range(x.shape[1] // 2):
//!         for j in range(x.shape[2] // 2):
//!             output[:, i, j, :] = np.max(x[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2, :], axis=(1, 2))
//!     return output
//!
//! def lenet5(input: Float[Array, '4 28 28 1'], conv1: Float[Array, '5 5 1 6'], conv1bias: Float[Array, '6'], conv2: Float[Array, '5 5 6 16'], conv2bias: Float[Array, '16'], fc1w: Float[Array, '256 120'], fc1b: Float[Array, '120'], fc2w: Float[Array, '120 84'], fc2b: Float[Array, '84'], fc3w: Float[Array, '84 10'], fc3b: Float[Array, '10'], N: Integer, C_before_fc1: Integer):
//!     x = relu(conv2d(input, conv1) + conv1bias)
//!     x = maxpool2d(x)
//!     x = relu(conv2d(x, conv2) + conv2bias)
//!     x = maxpool2d(x)
//!     x = np.reshape(x, (N, C_before_fc1))
//!     x = relu(x @ fc1w + fc1b)
//!     x = relu(x @ fc2w + fc2b)
//!     return x @ fc3w + fc3b
//! ```
//!
//! `conv2d` stores its sums as `float32`; Zinnia has one float type, so they
//! stay Q32 here.
use crate::prelude::*;

fn relu(x: Q32) -> Q32 {
    x.max(Q32::ZERO)
}

/// `relu(conv2d(input, weights) + bias)`: a valid `K x K` convolution whose
/// window products are multiplied one by one and summed (`np.sum` of an
/// element-wise product, not a contraction).
fn conv_relu(input: &Array4<Q32>, weights: &Array4<Q32>, bias: &Array1<Q32>) -> Array4<Q32> {
    let (n, h, w, c_in) = input.dim();
    let (k, kw, wc_in, c_out) = weights.dim();
    assert!(kw == k && wc_in == c_in, "operands could not be broadcast together");
    assert_eq!(bias.len(), c_out, "operands could not be broadcast together");
    let extent = |d: usize| (d + 1).checked_sub(k).expect("negative dimensions are not allowed");
    let (h_out, w_out) = (extent(h), extent(w));
    Array4::from_shape_fn((n, h_out, w_out, c_out), |(b, i, j, co)| {
        let mut sum = Q32::ZERO;
        for ki in 0..k {
            for kj in 0..k {
                for ci in 0..c_in {
                    sum += input[(b, i + ki, j + kj, ci)] * weights[(ki, kj, ci, co)];
                }
            }
        }
        relu(sum + bias[co])
    })
}

/// 2x2 max-pooling with stride 2 (an odd last row or column is dropped).
fn maxpool2d(x: &Array4<Q32>) -> Array4<Q32> {
    let (n, h, w, c) = x.dim();
    Array4::from_shape_fn((n, h / 2, w / 2, c), |(b, i, j, ch)| {
        let window = [(0, 0), (0, 1), (1, 0), (1, 1)];
        window
            .iter()
            .map(|&(di, dj)| x[(b, 2 * i + di, 2 * j + dj, ch)])
            .max()
            .expect("a 2x2 window")
    })
}

/// `np.reshape`'s extents for the target shape `(rows, cols)` of `len`
/// elements; one of them may be -1, inferred from the other.
fn reshape_extents(len: usize, rows: i64, cols: i64) -> (usize, usize) {
    let extent = |d: i64| usize::try_from(d).expect("cannot reshape to a negative extent");
    let infer = |known: usize| {
        assert!(known != 0 && len % known == 0, "cannot reshape array");
        len / known
    };
    match (rows, cols) {
        (-1, -1) => panic!("can only specify one unknown dimension"),
        (-1, cols) => (infer(extent(cols)), extent(cols)),
        (rows, -1) => (extent(rows), infer(extent(rows))),
        (rows, cols) => (extent(rows), extent(cols)),
    }
}

/// `x @ w + b`.
fn dense(x: &Array2<Q32>, w: &Array2<Q32>, b: &Array1<Q32>) -> Array2<Q32> {
    let mut y = np::matmul(x.view(), w.view());
    assert_eq!(b.len(), y.ncols(), "operands could not be broadcast together");
    for mut row in y.rows_mut() {
        row += b;
    }
    y
}

pub mod lenet5 {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input: Array4<Q32> = args.array4();
        let conv1: Array4<Q32> = args.array4();
        let conv1bias: Array1<Q32> = args.array1();
        let conv2: Array4<Q32> = args.array4();
        let conv2bias: Array1<Q32> = args.array1();
        let fc1w: Array2<Q32> = args.array2();
        let fc1b: Array1<Q32> = args.array1();
        let fc2w: Array2<Q32> = args.array2();
        let fc2b: Array1<Q32> = args.array1();
        let fc3w: Array2<Q32> = args.array2();
        let fc3b: Array1<Q32> = args.array1();
        let n: i64 = args.scalar();
        let c_before_fc1: i64 = args.scalar();

        let x = maxpool2d(&conv_relu(&input, &conv1, &conv1bias));
        let x = maxpool2d(&conv_relu(&x, &conv2, &conv2bias));
        let shape = reshape_extents(x.len(), n, c_before_fc1);
        let x = Array2::from_shape_vec(shape, x.iter().copied().collect())
            .expect("cannot reshape array");
        let x = dense(&x, &fc1w, &fc1b).mapv(relu);
        let x = dense(&x, &fc2w, &fc2b).mapv(relu);
        dense(&x, &fc3w, &fc3b)
    }
}
