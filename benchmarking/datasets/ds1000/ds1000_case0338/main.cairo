//! DS-1000 case 0338: the diagonal and the anti-diagonal of a square matrix,
//! stacked as two rows.
//!
//!     def kernel(a: Float[Array, "8 8"]):
//!         result = np.vstack((np.diag(a), np.diag(np.fliplr(a))))
//!         return result
// output: [q32]
use crate::q32::Q32;

const N: u32 = 8;

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == N * N, "a has the wrong length");
    let mut result = array![];
    for i in 0..N {
        result.append(*a.at(i * N + i));
    }
    for i in 0..N {
        result.append(*a.at(i * N + (N - 1 - i)));
    }
    result
}
