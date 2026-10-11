//! Shape-generic Q32 port of zkvm.rs::md.
// output: tensor<1,c32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{C32, C32Trait};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let e: Array<C32> = args.array1();
    let symbols: Array<C32> = args.array1();
    let mut result = array![];
    for ei in e.span() {
        let mut best = *symbols.at(0);
        let magnitude = (*ei - best).abs();
        let mut best_dist = magnitude * magnitude;
        for j in 1..symbols.len() {
            let symbol = *symbols.at(j);
            let magnitude = (*ei - symbol).abs();
            let distance = magnitude * magnitude;
            if distance < best_dist {
                best = symbol;
                best_dist = distance;
            };
        }
        result.append(best);
    }
    let mut out = args.output();
    tensor(ref out, array![e.len()].span(), result.span());
    out
}
