//! Cronbach's alpha; each squared deviation is individually rescaled.
// output: q32
use crate::io::{ArgsTrait, Word};
use crate::q32::{Q32, Q32Trait, ZERO, ONE};
fn sample_variance(values: Span<Q32>) -> Q32 {
    let n: i64 = values.len().into();
    let mut sum = ZERO;
    for v in values { sum = sum + *v; };
    let mean = sum.div_int(n);
    let mut squares = ZERO;
    for v in values { let d = *v - mean; squares = squares + d * d; };
    squares.div_int(n - 1)
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, scores): (u32, u32, Array<Q32>) = args.array2();
    let mut itemvars = ZERO;
    for i in 0..rows { itemvars = itemvars + sample_variance(scores.span().slice(i * cols, cols)); };
    let mut totals = array![];
    for j in 0..cols {
        let mut sum = ZERO;
        for i in 0..rows { sum = sum + *scores.at(i * cols + j); };
        totals.append(sum);
    };
    let n: i64 = rows.into();
    assert!(n != 1, "division by zero");
    let result = Q32Trait::from_int(n).div_int(n - 1) * (ONE - itemvars / sample_variance(totals.span()));
    let mut out = args.output();
    Word::write(result, ref out);
    out
}
