//! DS-1000 case 0456: pairwise Euclidean distances.
// output: [q32]
use crate::q32::{Q32, Q32Trait, ZERO};

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == 8 * 3, "a has the wrong length");
    let mut out = array![];
    for i in 0_u32..8 {
        for j in 0_u32..8 {
            let mut squared = ZERO;
            for k in 0_u32..3 {
                let delta = *a.at(j * 3 + k) - *a.at(i * 3 + k);
                squared = squared + delta * delta;
            }
            out.append(squared.sqrt());
        }
    }
    out
}
