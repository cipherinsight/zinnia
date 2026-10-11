//! Add the sum of tanh over the diagonal to every matrix element.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r, c, a): (u32, u32, Array<Q32>) = args.array2();
    assert!(r <= c, "diagonal index out of bounds");
    let mut trace = ZERO;
    for i in 0..r {
        trace = trace + (*a.at(i * c + i)).tanh();
    }
    let mut result = array![];
    for v in a {
        result.append(v + trace);
    }
    let mut out = args.output();
    tensor(ref out, array![r, c].span(), result.span());
    out
}
