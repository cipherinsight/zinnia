// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ONE};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let alpha: Q32 = args.scalar();
    let mut result = array![];
    if !x.is_empty() {
        let mut previous = *x.at(0);
        result.append(previous);
        for i in 1..x.len() {
            previous = alpha * *x.at(i) + (ONE - alpha) * previous;
            result.append(previous);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![result.len()].span(), result.span());
    out
}
