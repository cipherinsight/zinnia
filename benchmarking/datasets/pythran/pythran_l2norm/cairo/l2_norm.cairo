//! Row norms; each square is rounded before summing, as in Rust.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, x): (u32, u32, Array<Q32>) = args.array2();
    let mut result = array![];
    for i in 0..rows {
        let mut sum = ZERO;
        for j in 0..cols { sum = sum + (*x.at(i * cols + j)).abs().powi(2); };
        result.append(sum.sqrt());
    };
    let mut out = args.output();
    tensor(ref out, array![rows].span(), result.span());
    out
}
