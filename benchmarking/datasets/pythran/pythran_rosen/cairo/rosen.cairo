// output: q32
use crate::io::{ArgsTrait, Word};
use crate::q32::{Q32, Q32Trait, ONE, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let mut result = ZERO;
    for i in 1..x.len() {
        let xi = *x.at(i - 1);
        let a = *x.at(i) - xi * xi;
        let b = ONE - xi;
        result = result + (a * a).mul_int(100) + b * b;
    };
    let mut out = args.output();
    Word::write(result, ref out);
    out
}
