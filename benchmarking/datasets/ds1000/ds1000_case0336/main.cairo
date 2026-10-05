//! DS-1000 case 0336: the anti-diagonal of a square matrix,
//! `np.diag(np.fliplr(a))`.
//!
//!     def kernel(a: Float[Array, "8 8"]):
//!         result = np.diag(np.fliplr(a))
//!         return result
// output: [q32]
use crate::q32::Q32;

const N: u32 = 8;

/// `np.fliplr` reverses every row, so entry `i` of its diagonal is
/// `a[i][N - 1 - i]`.
#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N * N, "a has the wrong length");
    let mut result = array![];
    for i in 0..N {
        result.append(*a.at(i * N + (N - 1 - i)));
    }
    result
}
