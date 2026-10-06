//! Pythran value_index: the rows of `score` ordered by their second column.
//!
//! ```python
//! def tri(score: Integer[Array, '10 10'], score_t: Integer[Array, '10 10']):
//!     score_t[0, 0] = score[0, 0]
//!     score_t = score[np.argsort(score[:, 1])]
//!     return score_t
//! ```
//!
//! The argument `score_t` is only written before the name is rebound, so its
//! values never reach the result. `np.argsort`'s default sort is not stable;
//! rows with equal keys keep their order here (stable sort), which NumPy
//! does not promise.
use crate::prelude::*;

pub mod tri {
    use super::*;

    // unused parameters: score_t
    pub type Output = Array2<i64>;

    pub fn main(args: &mut Args<impl Input>) -> Output {
        let score: Array2<i64> = args.array2();
        assert!(score.nrows() > 0, "score[0, 0]: index 0 is out of bounds");
        let keys: Vec<i64> = score.column(1).to_vec();
        score.select(Axis(0), &np::argsort(&keys))
    }
}
