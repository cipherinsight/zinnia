//! Rows sorted stably by column one, matching the Rust tie behavior.
// output: tensor<2,i64>
use crate::io::{ArgsTrait, tensor};
// unused parameters: score_t
use crate::np::argsort;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, score): (u32, u32, Array<i64>) = args.array2();
    assert!(rows > 0 && cols > 1, "score index out of bounds");
    let mut keys = array![];
    for i in 0..rows { keys.append(*score.at(i * cols + 1)); };
    let order = argsort(keys.span());
    let mut result = array![];
    for i in order { result.append_span(score.span().slice(i * cols, cols)); };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
