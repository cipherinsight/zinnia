//! Pythran weights: sequence weights for a multiple alignment, the inverse of
//! the number of sequences that agree with each sequence at more than
//! `threshold * length` positions.
//!
//! ```python
//! def weights(input_data: Integer[Array, '5 2'], threshold: Float=0.3):
//!     n_seq, length = input_data.shape
//!     weights = np.zeros(n_seq, dtype=np.float32)
//!     for i in range(n_seq):
//!         vector = input_data[i, None, :]
//!         count_matches = np.sum(vector == input_data, axis=1)
//!         over_threshold = count_matches > threshold * length
//!         total = np.sum(over_threshold)
//!         weights[i] = 1 / total
//!     return weights
//! ```
//!
//! NumPy stores the weights as float32; here they are Q32.
use crate::prelude::*;

pub mod weights {
    use super::*;

    pub type Output = Array1<Q32>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let input_data: Array2<i64> = args.array2();
        let threshold: Q32 = args.scalar();
        let length = input_data.ncols() as i64;
        let bound = threshold * length;
        input_data
            .rows()
            .into_iter()
            .map(|vector| {
                let total = input_data
                    .rows()
                    .into_iter()
                    .filter(|other| {
                        let count_matches =
                            vector.iter().zip(other).filter(|(a, b)| a == b).count() as i64;
                        Q32::from_int(count_matches) > bound
                    })
                    .count() as i64;
                Q32::ONE / total
            })
            .collect()
    }
}
