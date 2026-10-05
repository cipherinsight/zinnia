//! DS-1000 case 0353: the rows of `A` that do not occur in `B`. Each row is
//! turned into one integer with `np.ravel_multi_index` over a grid just large
//! enough for both arrays, and the integers are compared with `np.in1d`.
//!
//!     def kernel(A: Integer[Array, "2 3"], B: Integer[Array, "7 3"]):
//!         dims = np.maximum(B.max(0),A.max(0))+1
//!         output = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
//!         return output
// output: [i64]
use core::cmp::max;
use crate::np::contains;

const A_ROWS: u32 = 2;
const B_ROWS: u32 = 7;
const COLS: u32 = 3;

/// The kept rows of `A`, flattened row-major (shape `(k, 3)`).
#[executable]
fn main(A: Array<i64>, B: Array<i64>) -> Array<i64> {
    assert!(A.len() == A_ROWS * COLS && B.len() == B_ROWS * COLS, "inputs have the wrong shape");
    let (a, b) = (A.span(), B.span());
    let dims = grid_dims(a, b);
    let mut b_keys = array![];
    for r in 0..B_ROWS {
        b_keys.append(ravel_multi_index(b.slice(r * COLS, COLS), dims));
    }
    let b_keys = b_keys.span();
    let mut kept = array![];
    for r in 0..A_ROWS {
        let row = a.slice(r * COLS, COLS);
        if !contains(b_keys, ravel_multi_index(row, dims)) {
            kept.append_span(row);
        }
    }
    kept
}

/// `np.maximum(B.max(0), A.max(0)) + 1`: one more than the largest entry of
/// each column. Like NumPy, refuses a grid whose size overflows an `int64`
/// (the `i64` product panics on overflow).
fn grid_dims(a: Span<i64>, b: Span<i64>) -> Span<i64> {
    let mut dims = array![];
    let mut size: i64 = 1;
    for c in 0..COLS {
        let mut largest = *a.at(c);
        for r in 1..A_ROWS {
            largest = max(largest, *a.at(r * COLS + c));
        }
        for r in 0..B_ROWS {
            largest = max(largest, *b.at(r * COLS + c));
        }
        size = size * (largest + 1);
        dims.append(largest + 1);
    }
    dims.span()
}

/// `np.ravel_multi_index(row, dims)` in C order; like NumPy, fails on a
/// coordinate outside `[0, dim)`.
fn ravel_multi_index(row: Span<i64>, dims: Span<i64>) -> i64 {
    let mut flat = 0;
    for c in 0..COLS {
        let (coord, dim) = (*row.at(c), *dims.at(c));
        assert!(0 <= coord && coord < dim, "invalid entry in coordinates array");
        flat = flat * dim + coord;
    }
    flat
}
