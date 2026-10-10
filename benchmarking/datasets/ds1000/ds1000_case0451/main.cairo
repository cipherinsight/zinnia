//! DS-1000 case 0451: np.zeros((20, 10, 10, 2)).
// output: [q32]
use crate::q32::{Q32, ZERO};

#[executable]
fn main() -> Array<Q32> {
    let mut arr = array![];
    for _ in 0_u32..20 * 10 * 10 * 2 {
        arr.append(ZERO);
    }
    arr
}
