//! DS-1000 case 0364: insert a row into a matrix,
//! `np.insert(a, pos, element, axis=0)`.
//!
//!     def kernel(a: Float[Array, "100 10"], pos: Integer, element: Float[Array, "10"]):
//!         a = np.insert(a, pos, element, axis = 0)
//!         return a
// output: [q32]
use crate::q32::Q32;

const ROWS: u32 = 100;
const COLS: u32 = 10;

#[executable]
fn main(a: Array<Q32>, pos: i64, element: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS && element.len() == COLS, "inputs have the wrong length");
    let at = insert_position(pos, ROWS) * COLS;
    let a = a.span();
    let mut result = array![];
    result.append_span(a.slice(0, at));
    result.append_span(element.span());
    result.append_span(a.slice(at, ROWS * COLS - at));
    result
}

/// `np.insert`'s position on an axis of length `len`: anything in
/// `[-len, len]`, negative positions counting from the end; NumPy raises
/// `IndexError` outside that range.
fn insert_position(pos: i64, len: u32) -> u32 {
    let n: i64 = len.into();
    assert!(-n <= pos && pos <= n, "insert position out of range");
    let wrapped = if pos < 0 {
        pos + n
    } else {
        pos
    };
    wrapped.try_into().unwrap()
}
