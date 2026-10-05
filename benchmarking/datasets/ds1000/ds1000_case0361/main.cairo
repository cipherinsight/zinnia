//! DS-1000 case 0361: drop the first and third columns,
//! `np.delete(a, [0, 2], axis=1)`.
//!
//!     def kernel(a: Float[Array, "6 6"]):
//!         temp = np.array([0, 2])
//!         a = np.delete(a, temp, axis = 1)
//!         return a
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 6;
const COLS: u32 = 6;
const DELETED: [u32; 2] = [0, 2];

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS, "a has the wrong length");
    let kept = kept_columns(DELETED.span());
    let mut result = array![];
    for r in 0..ROWS {
        for j in kept {
            result.append(*a.at(r * COLS + *j));
        }
    }
    result
}

/// The columns `np.delete` keeps, in order.
fn kept_columns(deleted: Span<u32>) -> Span<u32> {
    let mut kept = array![];
    for j in 0..COLS {
        let mut is_deleted = false;
        for d in deleted {
            if *d == j {
                is_deleted = true;
            }
        }
        if !is_deleted {
            kept.append(j);
        }
    }
    kept.span()
}
