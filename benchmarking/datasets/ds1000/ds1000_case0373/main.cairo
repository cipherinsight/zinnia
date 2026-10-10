//! DS-1000 case 0373: sort the grades, then return ECDF probabilities.
// output: [q32]
use crate::np::sort;
use crate::q32::{Q32, Q32Trait};

#[executable]
fn main(grades: Array<Q32>) -> Array<Q32> {
    assert!(grades.len() == 50, "grades has the wrong length");
    // The kernel sorts even though only the sorted array's length is used.
    let xs = sort(grades.span());
    let n = xs.len();
    let denominator = Q32Trait::from_int(n.into());
    let mut ys = array![];
    for rank in 1..n + 1 {
        ys.append(Q32Trait::from_int(rank.into()) / denominator);
    }
    ys
}
