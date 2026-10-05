//! NPBench deep_learning/mlp: a three-layer perceptron (ReLU, ReLU, softmax
//! over the last axis).
//!
//! ```python
//! def relu(x):
//!     return np.maximum(x, 0)
//!
//! def softmax(x):
//!     tmp_max = np.max(x, axis=-1, keepdims=True)
//!     tmp_out = np.exp(x - tmp_max)
//!     tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
//!     return tmp_out / tmp_sum
//!
//! def mlp(input: Float[Array, '8 3'], w1: Float[Array, '3 30000'], b1: Float[Array, '30000'], w2: Float[Array, '30000 2000'], b2: Float[Array, '2000'], w3: Float[Array, '2000 2000'], b3: Float[Array, '2000']):
//!     x = relu(input @ w1 + b1)
//!     x = relu(x @ w2 + b2)
//!     x = softmax(x @ w3 + b3)
//!     return x
//! ```
use crate::prelude::*;

fn relu(x: Q32) -> Q32 {
    x.max(Q32::ZERO)
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

/// Softmax of each row, shifted by the row maximum.
fn softmax(mut x: Array2<Q32>) -> Array2<Q32> {
    for mut row in x.rows_mut() {
        let max = *row.iter().max().expect("zero-size array to reduction operation maximum");
        row.mapv_inplace(|v| (v - max).exp());
        let sum: Q32 = row.sum();
        row.mapv_inplace(|e| e / sum);
    }
    x
}

pub mod mlp {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input: Array2<Q32> = args.array2();
        let w1: Array2<Q32> = args.array2();
        let b1: Array1<Q32> = args.array1();
        let w2: Array2<Q32> = args.array2();
        let b2: Array1<Q32> = args.array1();
        let w3: Array2<Q32> = args.array2();
        let b3: Array1<Q32> = args.array1();
        let x = dense(&input, &w1, &b1).mapv(relu);
        let x = dense(&x, &w2, &b2).mapv(relu);
        softmax(dense(&x, &w3, &b3))
    }
}
