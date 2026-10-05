//! DS-1000 case 0485: `np.tensordot(A, B, axes=((2), (0)))`, the last axis of
//! `A` contracted with the first axis of `B`.
//!
//!     def kernel(A: Float[Array, "5 6 3"], B: Float[Array, "3 3"]):
//!         result = np.tensordot(A,B,axes=((2),(0)))
//!         return result
// output: [q32]
use crate::q32::{Q32, from_raw_product, raw_product};

const ROWS: u32 = 5 * 6;
const K: u32 = 3;
const COLS: u32 = 3;

/// Each output is a dot product, accumulated unscaled and rescaled once.
#[executable]
fn main(A: Array<Q32>, B: Array<Q32>) -> Array<Q32> {
    assert!(A.len() == ROWS * K && B.len() == K * COLS, "inputs have the wrong shape");
    let mut out = array![];
    for r in 0..ROWS {
        for j in 0..COLS {
            let mut acc: i128 = 0;
            for k in 0..K {
                acc += raw_product(*A.at(r * K + k), *B.at(k * COLS + j));
            }
            out.append(from_raw_product(acc));
        }
    }
    out
}
