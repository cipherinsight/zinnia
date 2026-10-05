//! DS-1000 case 0481: every index `i` with `(x[i], y[i]) == (a, b)`.
//!
//!     def kernel(x: Integer[Array, "9"], y: Integer[Array, "9"], a: Integer, b: Integer):
//!         idx_list = ((x == a) & (y == b))
//!         result = idx_list.nonzero()[0]
//!         return result
// output: [i64]
const N: u32 = 9;

#[executable]
fn main(x: Array<i64>, y: Array<i64>, a: i64, b: i64) -> Array<i64> {
    assert!(x.len() == N && y.len() == N, "inputs have the wrong length");
    let mut indices = array![];
    for i in 0..N {
        if *x.at(i) == a && *y.at(i) == b {
            indices.append(i.into());
        }
    }
    indices
}
