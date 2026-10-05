//! DS-1000 case 0362: delete the columns listed 1-based in `del_col`,
//! ignoring numbers past the last column.
//!
//!     def kernel(a: Float[Array, "8 6"]):
//!         del_col = np.array([12, 14, 1, 9])
//!         mask = (del_col <= a.shape[1])
//!         del_col = del_col[mask] - 1
//!         result = np.delete(a, del_col, axis=1)
//!         return result
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;
/// `del_col` as the kernel sets it: 1-based column numbers.
const DEL_COL: [u32; 4] = [12, 14, 1, 9];

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let kept = kept_columns();
    let mut result = array![];
    for r in 0..ROWS {
        for j in kept {
            result.append(*a.at(r * COLS + *j));
        }
    }
    result
}

/// The columns `np.delete` keeps, in order: those that
/// `del_col[del_col <= COLS] - 1` does not list (the mask only drops numbers
/// that match no column).
fn kept_columns() -> Span<u32> {
    let mut kept = array![];
    for j in 0..COLS {
        let mut is_deleted = false;
        for d in DEL_COL.span() {
            if *d == j + 1 {
                is_deleted = true;
            }
        }
        if !is_deleted {
            kept.append(j);
        }
    }
    kept.span()
}
