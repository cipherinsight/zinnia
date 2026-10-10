//! DS-1000 case 0413: keep complex entries with nonzero imaginary part.
// Complex values are represented by (real, imaginary) Q32 pairs.
// output: [(q32, q32)]
use crate::q32::{Q32, ZERO};

#[executable]
fn main(x: Array<(Q32, Q32)>) -> Array<(Q32, Q32)> {
    assert!(x.len() == 10, "x has the wrong length");
    let mut kept = array![];
    for v in x {
        if v.1 != ZERO {
            kept.append(v);
        }
    }
    kept
}
