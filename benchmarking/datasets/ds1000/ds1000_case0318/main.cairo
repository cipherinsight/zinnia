//! DS-1000 case 0318: reorder the columns of `a` by the inverse of
//! `permutation`.
//!
//!     def kernel(a: Float[Array, "8 6"], permutation: Integer[Array, "6"]):
//!         c = np.empty_like(permutation)
//!         c[permutation] = np.arange(len(permutation))
//!         a = a[:, c]
//!         return a
// output: [q32]
use crate::np::inverse_permutation;
use crate::q32::Q32;

const ROWS: u32 = 8;
const COLS: u32 = 6;

#[executable]
fn main(a: Array<Q32>, permutation: Array<i64>) -> Array<Q32> {
    assert!(a.len() == ROWS * COLS && permutation.len() == COLS, "inputs have the wrong length");
    let c = inverse_permutation(permutation.span()).span();
    let mut result = array![];
    for r in 0..ROWS {
        for j in c {
            result.append(*a.at(r * COLS + *j));
        }
    }
    result
}
