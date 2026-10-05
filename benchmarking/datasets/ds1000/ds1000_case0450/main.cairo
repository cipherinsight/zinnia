//! DS-1000 case 0450: `a[:, second.reshape(-1, 1), third]`. The two index
//! arrays broadcast to shape `(2, 2)` and replace axes 1 and 2, so the result
//! has shape `(7, 2, 2)`: `result[i][p][q] = a[i][second[p]][third[q]]`.
//!
//!     def kernel(a: Float[Array, "7 8 9"], second: Integer[Array, "2"], third: Integer[Array, "2"]):
//!         result = a[:, np.array(second).reshape(-1,1), third]
//!         return result
// output: [q32]
use crate::np::np_index;
use crate::q32::Q32;

const PLANES: u32 = 7;
const ROWS: u32 = 8;
const COLS: u32 = 9;
const PICKS: u32 = 2;

#[executable]
fn main(a: Array<Q32>, second: Array<i64>, third: Array<i64>) -> Array<Q32> {
    assert!(a.len() == PLANES * ROWS * COLS, "a has the wrong length");
    assert!(second.len() == PICKS && third.len() == PICKS, "index arrays have the wrong length");
    let rows = wrap_indices(second.span(), ROWS);
    let cols = wrap_indices(third.span(), COLS);
    let mut out = array![];
    for plane in 0..PLANES {
        for r in rows.span() {
            for c in cols.span() {
                out.append(*a.at((plane * ROWS + *r) * COLS + *c));
            }
        }
    }
    out
}

/// Every index of `indices` checked and wrapped into an axis of length `len`.
fn wrap_indices(indices: Span<i64>, len: u32) -> Array<u32> {
    let mut wrapped = array![];
    for i in indices {
        wrapped.append(np_index(*i, len));
    }
    wrapped
}
