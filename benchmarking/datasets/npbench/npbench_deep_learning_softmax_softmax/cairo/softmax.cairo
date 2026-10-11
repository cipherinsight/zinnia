//! Softmax over the final axis of a dynamic four-dimensional tensor.
// output: tensor<4,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (a, b, c, d, x): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let result = normalize(x.span(), a * b * c, d);
    let mut out = args.output();
    tensor(ref out, array![a, b, c, d].span(), result.span());
    out
}

// Normalize each contiguous lane, preserving the exp/sum/divide order.
fn normalize(x: Span<Q32>, rows: u32, cols: u32) -> Array<Q32> {
    let mut result = array![];
    for i in 0..rows {
        assert!(cols > 0, "maximum of an empty lane");
        let lane = x.slice(i * cols, cols);
        let mut maximum = *lane.at(0);
        for v in lane { if *v > maximum { maximum = *v; }; };
        let mut exps = array![];
        let mut sum = ZERO;
        for v in lane {
            let e = (*v - maximum).exp();
            exps.append(e);
            sum = sum + e;
        };
        for e in exps { result.append(e / sum); };
    };
    result
}
