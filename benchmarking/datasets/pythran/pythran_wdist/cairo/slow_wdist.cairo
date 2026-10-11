//! Weighted column distances, with NumPy broadcasting of each column operation.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};

fn broadcast(a: u32, b: u32) -> u32 {
    if a == b { a } else if a == 1 { b } else { assert!(b == 1, "broadcast mismatch"); a }
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    let (wr, wc, w): (u32, u32, Array<Q32>) = args.array2();
    let mut result = array![];
    for i in 0..ac {
        for j in 0..bc {
            let dr = broadcast(ar, br);
            assert!(i < wc, "W column out of bounds");
            let rows = broadcast(dr, wr);
            let mut sum = ZERO;
            for k in 0..rows {
                let di = if dr == 1 { 0 } else { k };
                let ai = if ar == 1 { 0 } else { di };
                let bi = if br == 1 { 0 } else { di };
                let wi = if wr == 1 { 0 } else { k };
                let d = (*a.at(ai * ac + i) - *b.at(bi * bc + j)) / *w.at(wi * wc + i);
                sum = sum + d * d;
            };
            result.append(sum.sqrt());
        };
    };
    let mut out = args.output();
    tensor(ref out, array![ac, bc].span(), result.span());
    out
}
