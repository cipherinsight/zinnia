//! Suffix sums, accumulated from the final element as in Rust.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, Word};
use crate::q32::{Q32, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let n = x.len();
    let mut reversed = array![];
    let mut total = ZERO;
    let mut i = n;
    while i > 0 {
        i -= 1;
        total = total + *x.at(i);
        reversed.append(total);
    };
    let mut out = args.output();
    out.append(n.into());
    let mut i = n;
    while i > 0 { i -= 1; Word::write(*reversed.at(i), ref out); };
    out
}
