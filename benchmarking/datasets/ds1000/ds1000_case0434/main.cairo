//! DS-1000 case 0434: zero the listed rows and columns of a 4x4 matrix.
//!
//!     def kernel(a: Integer[Array, "4 4"], zero_rows: Integer[Array, "2"], zero_cols: Integer[Array, "2"]):
//!         a[zero_rows, :] = 0
//!         a[:, zero_cols] = 0
//!         return a
// output: [i64]
use crate::np::np_index;

const N: u32 = 4;
const K: u32 = 2;

#[executable]
fn main(a: Array<i64>, zero_rows: Array<i64>, zero_cols: Array<i64>) -> Array<i64> {
    assert!(a.len() == N * N, "a has the wrong length");
    assert!(zero_rows.len() == K && zero_cols.len() == K, "index arrays have the wrong length");
    let row_zeroed = selected(zero_rows.span());
    let col_zeroed = selected(zero_cols.span());
    let mut out = array![];
    for i in 0..N {
        for j in 0..N {
            out.append(if *row_zeroed.at(i) || *col_zeroed.at(j) {
                0
            } else {
                *a.at(i * N + j)
            });
        }
    }
    out
}

/// For each of the `N` positions of an axis, whether one of the
/// (Python-style) `indices` selects it.
fn selected(indices: Span<i64>) -> Span<bool> {
    let mut wrapped = array![];
    for i in indices {
        wrapped.append(np_index(*i, N));
    }
    let mut hit = array![];
    for p in 0..N {
        let mut found = false;
        for w in wrapped.span() {
            if *w == p {
                found = true;
            }
        }
        hit.append(found);
    }
    hit.span()
}
