//! DS-1000 case 0477: np.diag of a four-element float vector.
// output: [q32]
use crate::q32::{Q32, ZERO};

#[executable]
fn main(i: Array<Q32>) -> Array<Q32> {
    assert!(i.len() == 4, "i has the wrong length");
    let mut result = array![];
    for row in 0_u32..4 {
        for col in 0_u32..4 {
            result.append(if row == col { *i.at(row) } else { ZERO });
        }
    }
    result
}
