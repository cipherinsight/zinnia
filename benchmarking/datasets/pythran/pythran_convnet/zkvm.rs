//! Pythran convnet: a quasi-recurrent (QRNN) layer: a matrix product gives
//! the candidate, forget and output gates of every step, and fo-pooling runs
//! forwards and backwards over the sequence.
//!
//! ```python
//! def sigmoid(z):
//!     return 1 / (1 + np.exp(-z))
//!
//! def convnet(conv_matrix: Float[Array, '4 4'], qcnn_filter: Float[Array, '4 4'], length: Integer, batch_size: Integer, state_size: Integer):
//!     state_size_2 = state_size * 2
//!     state_size_3 = state_size * 3
//!     state_size_4 = state_size * 4
//!     conv_results = np.dot(conv_matrix, qcnn_filter).reshape(length, batch_size, state_size_4)
//!     conv_results[:, :, :state_size] = np.tanh(conv_results[:, :, :state_size])
//!     conv_results[:, :, state_size:state_size_4] = sigmoid(conv_results[:, :, state_size:state_size_4])
//!     state_results = np.zeros((batch_size, length, state_size_2), dtype=np.float32)
//!     state = np.zeros((batch_size, state_size), dtype=np.float32)
//!     for i in range(length):
//!         z = conv_results[i, :, :state_size]
//!         f = conv_results[i, :, state_size:state_size_2]
//!         o = conv_results[i, :, state_size_2:state_size_3]
//!         state = f * state + (1 - f) * z
//!         state_results[:, i, :state_size] = state * o
//!     state = np.zeros((batch_size, state_size), dtype=np.float32)
//!     for i in range(length - 1, -1, -1):
//!         z = conv_results[i, :, :state_size]
//!         f = conv_results[i, :, state_size_3:state_size_4]
//!         o = conv_results[i, :, state_size_2:state_size_3]
//!         state = f * state + (1 - f) * z
//!         state_results[:, i, state_size:] = state * o
//!     state_results = state_results.reshape((batch_size * length, state_size_2))
//!     return state_results
//! ```
//!
//! `state_results` is float32 in NumPy; here it is Q32 like every float.
use crate::prelude::*;

/// `sigmoid(z) = 1 / (1 + exp(-z))`.
fn sigmoid(z: Q32) -> Q32 {
    Q32::ONE / (Q32::ONE + (-z).exp())
}

/// One fo-pooling step on a batch row: `state = f * state + (1 - f) * z`,
/// returning `state * o`.
fn pool(state: &mut Q32, z: Q32, f: Q32, o: Q32) -> Q32 {
    *state = f * *state + (Q32::ONE - f) * z;
    *state * o
}

pub mod convnet {
    use super::*;

    pub type Output = Array2<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let conv_matrix: Array2<Q32> = args.array2();
        let qcnn_filter: Array2<Q32> = args.array2();
        let length: i64 = args.scalar();
        let batch_size: i64 = args.scalar();
        let state_size: i64 = args.scalar();
        let dim = |d: i64| usize::try_from(d).expect("negative dimensions are not allowed");
        let (length, batch, ss) = (dim(length), dim(batch_size), dim(state_size));

        let mut conv = np::matmul(conv_matrix.view(), qcnn_filter.view())
            .into_shape_with_order((length, batch, 4 * ss))
            .expect("cannot reshape the product into (length, batch_size, 4 * state_size)");
        // Gates: z = tanh on [0, ss), sigmoid on f, o and the backward f.
        for mut gates in conv.lanes_mut(Axis(2)) {
            for (c, g) in gates.iter_mut().enumerate() {
                *g = if c < ss { g.tanh() } else { sigmoid(*g) };
            }
        }

        let mut state_results = Array3::<Q32>::zeros((batch, length, 2 * ss));
        let mut state = Array2::<Q32>::zeros((batch, ss));
        for i in 0..length {
            for b in 0..batch {
                for s in 0..ss {
                    let gate = |offset: usize| conv[(i, b, offset + s)];
                    state_results[(b, i, s)] = pool(&mut state[(b, s)], gate(0), gate(ss), gate(2 * ss));
                }
            }
        }
        state.fill(Q32::ZERO);
        for i in (0..length).rev() {
            for b in 0..batch {
                for s in 0..ss {
                    let gate = |offset: usize| conv[(i, b, offset + s)];
                    state_results[(b, i, ss + s)] = pool(&mut state[(b, s)], gate(0), gate(3 * ss), gate(2 * ss));
                }
            }
        }
        state_results
            .into_shape_with_order((batch * length, 2 * ss))
            .expect("a contiguous array reshapes")
    }
}
