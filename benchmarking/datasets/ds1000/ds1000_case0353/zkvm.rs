//! DS-1000 case 0353: the rows of `A` that do not occur in `B`. Each row is
//! turned into one integer with `np.ravel_multi_index` over a grid just large
//! enough for both arrays, and the integers are compared with `np.in1d`.
//!
//! ```python
//! def kernel(A: Integer[Array, "2 3"], B: Integer[Array, "7 3"]):
//!     dims = np.maximum(B.max(0),A.max(0))+1
//!     output = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
//!     return output
//! ```
use crate::prelude::*;

const A_ROWS: usize = 2;
const B_ROWS: usize = 7;
const COLS: usize = 3;

type Row = [i64; COLS];

/// The kept rows of `A`, flattened row-major (shape `(k, 3)`).
pub type Output = Vec<i64>;

pub fn main(input: &mut impl Input) -> Output {
    let a: [Row; A_ROWS] = input.read_matrix();
    let b: [Row; B_ROWS] = input.read_matrix();
    let dims = grid_dims(a.iter().chain(&b));
    let b_keys: Vec<i64> = b.iter().map(|row| ravel_multi_index(row, &dims)).collect();
    a.iter()
        .filter(|row| !b_keys.contains(&ravel_multi_index(row, &dims)))
        .flatten()
        .copied()
        .collect()
}

/// `np.maximum(B.max(0), A.max(0)) + 1`: one more than the largest entry of
/// each column. Like NumPy, refuses a grid whose size overflows an `int64`.
fn grid_dims<'a>(rows: impl Iterator<Item = &'a Row>) -> Row {
    let mut largest = [i64::MIN; COLS];
    for row in rows {
        for (m, &v) in largest.iter_mut().zip(row) {
            *m = (*m).max(v);
        }
    }
    let dims = largest.map(|m| m + 1);
    let size = dims.iter().try_fold(1_i64, |size, &d| size.checked_mul(d));
    assert!(size.is_some(), "invalid dims: array size defined by dims is too large");
    dims
}

/// `np.ravel_multi_index(row, dims)` in C order; like NumPy, panics on a
/// coordinate outside `[0, dim)`.
fn ravel_multi_index(row: &Row, dims: &Row) -> i64 {
    row.iter().zip(dims).fold(0, |flat, (&c, &d)| {
        assert!(0 <= c && c < d, "invalid entry in coordinates array");
        flat * d + c
    })
}
