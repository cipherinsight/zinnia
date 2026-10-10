//! DS-1000 case 0330: elementwise sixth power.
// output: [q32]
use crate::q32::{Q32, ONE};

#[executable]
fn main(a: Array<Q32>) -> Array<Q32> {
    assert!(a.len() == 8 * 6, "a has the wrong length");
    let mut out = array![];
    for v in a {
        let mut power = ONE;
        for _ in 0_u32..6 {
            power = power * v;
        }
        out.append(power);
    }
    out
}
