//! Piecewise tax; constants and per-segment sums match the Rust reference.
// output: q32
use crate::io::{ArgsTrait, Word};
use crate::q32::{Q32, Q32Trait, ZERO};
const RATE1: Q32 = Q32 { bits: 1932735283 };
const OFFSET1: Q32 = Q32 { bits: 69426127705211 };
const RATE2: Q32 = Q32 { bits: 1803886264 };
const OFFSET2: Q32 = Q32 { bits: 36401737619210 };
const SLOPE3: Q32 = Q32 { bits: 9610 };
const BASE3: Q32 = Q32 { bits: 1029503661 };
const OFFSET3: Q32 = Q32 { bits: 4035422422303 };
const SLOPE4: Q32 = Q32 { bits: 43262 };
const BASE4: Q32 = Q32 { bits: 601295421 };
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let d: Array<Q32> = args.array1();
    let (mut s1, mut s2, mut s3, mut s4, mut total) = (ZERO, ZERO, ZERO, ZERO, ZERO);
    for v in d {
        total = total + v;
        if v > Q32Trait::from_int(256303) { s1 = s1 + (v * RATE1 - OFFSET1); }
        else if v > Q32Trait::from_int(54057) { s2 = s2 + (v * RATE2 - OFFSET2); }
        else if v > Q32Trait::from_int(13769) {
            let seg = v - Q32Trait::from_int(13769);
            s3 = s3 + (seg * (seg * SLOPE3 + BASE3) + OFFSET3);
        } else if v > Q32Trait::from_int(8820) {
            let seg = v - Q32Trait::from_int(8820);
            s4 = s4 + seg * (seg * SLOPE4 + BASE4);
        };
    };
    let mut out = args.output();
    Word::write((s1 + s2 + s3 + s4) / total, ref out);
    out
}
