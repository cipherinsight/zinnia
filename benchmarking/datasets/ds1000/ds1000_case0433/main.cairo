//! DS-1000 case 0433: zero one row and one column of a 4x4 matrix.
//!
//!     def kernel(a: Integer[Array, "4 4"], zero_rows: Integer, zero_cols: Integer):
//!         a[zero_rows, :] = 0
//!         a[:, zero_cols] = 0
//!         return a
// output: [i64]
use crate::np::np_index;

const N: u32 = 4;

#[executable]
fn main(a: Array<i64>, zero_rows: i64, zero_cols: i64) -> Array<i64> {
    assert!(a.len() == N * N, "a has the wrong length");
    let zero_row = np_index(zero_rows, N);
    let zero_col = np_index(zero_cols, N);
    let mut out = array![];
    for i in 0..N {
        for j in 0..N {
            out.append(if i == zero_row || j == zero_col {
                0
            } else {
                *a.at(i * N + j)
            });
        }
    }
    out
}
