//! Hastings-Powell food-web derivative with the Rust Q32 evaluation order.
// output: tensor<1,q32>
// unused parameters: t
use crate::io::{ArgsTrait, tensor};
use crate::q32::{ONE, Q32};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let y: Array<Q32> = args.array1();
    let a1: Q32 = args.scalar();
    let a2: Q32 = args.scalar();
    let b1: Q32 = args.scalar();
    let b2: Q32 = args.scalar();
    let d1: Q32 = args.scalar();
    let d2: Q32 = args.scalar();
    let (y0, y1, y2) = (*y.at(0), *y.at(1), *y.at(2));
    let prey = a1 * y0 * y1 / (ONE + b1 * y0);
    let predator = a2 * y1 * y2 / (ONE + b2 * y1);
    let result = array![y0 * (ONE - y0) - prey, prey - predator - d1 * y1, predator - d2 * y2];
    let mut out = args.output();
    tensor(ref out, array![3].span(), result.span());
    out
}
