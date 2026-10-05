//! DS-1000 case 0450: `a[:, second.reshape(-1, 1), third]`. The two index
//! arrays broadcast to shape `(2, 2)` and replace axes 1 and 2, so the result
//! has shape `(7, 2, 2)`: `result[i][p][q] = a[i][second[p]][third[q]]`.
//!
//! ```python
//! def kernel(a: Float[Array, "7 8 9"], second: Integer[Array, "2"], third: Integer[Array, "2"]):
//!     result = a[:, np.array(second).reshape(-1,1), third]
//!     return result
//! ```
use crate::prelude::*;

const PLANES: usize = 7;
const ROWS: usize = 8;
const COLS: usize = 9;
const PICKS: usize = 2;

pub type Output = Vec<Q32>;

pub fn main(input: &mut impl Input) -> Output {
    let a: Vec<Q32> = input.read_vec(PLANES * ROWS * COLS);
    let second: [i64; PICKS] = input.read_array();
    let third: [i64; PICKS] = input.read_array();
    let rows = second.map(|i| np_index(i, ROWS));
    let cols = third.map(|i| np_index(i, COLS));
    let mut out = Vec::with_capacity(PLANES * PICKS * PICKS);
    for plane in a.chunks_exact(ROWS * COLS) {
        for &r in &rows {
            for &c in &cols {
                out.push(plane[r * COLS + c]);
            }
        }
    }
    out
}
