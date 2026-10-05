//! DS-1000 case 0435: zero row 1 and column 0 of a 4x4 matrix.
//!
//!     def kernel(a: Integer[Array, "4 4"]):
//!         a[1, :] = 0
//!         a[:, 0] = 0
//!         return a
// output: [i64]
const N: u32 = 4;

#[executable]
fn main(a: Array<i64>) -> Array<i64> {
    assert!(a.len() == N * N, "a has the wrong length");
    let mut out = array![];
    for i in 0..N {
        for j in 0..N {
            out.append(if i == 1 || j == 0 {
                0
            } else {
                *a.at(i * N + j)
            });
        }
    }
    out
}
