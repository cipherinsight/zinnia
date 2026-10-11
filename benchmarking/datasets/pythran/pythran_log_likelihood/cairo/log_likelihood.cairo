//! Normal-distribution log likelihood, preserving every Rust Q32 operation.
// output: q32
use crate::io::{ArgsTrait, Word};
use crate::q32::{Q32, Q32Trait, ZERO};
const SQRT_2PI: Q32 = Q32 { bits: 10765886463 };

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let data: Array<Q32> = args.array1();
    let mean: Q32 = args.scalar();
    let sigma: Q32 = args.scalar();
    assert!(sigma != ZERO, "requires sigma != 0.0");
    let two_var = (sigma * sigma).mul_int(2);
    let normalizer = SQRT_2PI * sigma;
    let mut result = ZERO;
    for x in data {
        let s = (x - mean) * (x - mean) / two_var;
        result = result + ((-s).exp() / normalizer).ln();
    };
    let mut out = args.output();
    Word::write(result, ref out);
    out
}
