// output: (tensor<1,q32>,tensor<1,bool>)
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let y: Array<bool> = args.array1();
    assert!(y.len() == 1, "ambiguous truth value");
    let truth = *y.at(0);
    let mut result = array![];
    for v in x { result.append(if truth { v } else { v + v }); };
    let mut out = args.output();
    tensor(ref out, array![result.len()].span(), result.span());
    tensor(ref out, array![y.len()].span(), y.span());
    out
}
