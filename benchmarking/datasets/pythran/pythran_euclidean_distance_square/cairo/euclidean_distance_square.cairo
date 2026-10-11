//! Gram contraction rounds once; the row square sums round each square.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
use crate::np::inner;
fn squares(x: Span<Q32>, rows: u32, cols: u32) -> Array<Q32> {
    let mut result = array![];
    for i in 0..rows {
        let mut sum = ZERO;
        for j in 0..cols { let v = *x.at(i * cols + j); sum = sum + v * v; };
        result.append(sum);
    };
    result
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r1, c1, x1): (u32, u32, Array<Q32>) = args.array2();
    let (r2, c2, x2): (u32, u32, Array<Q32>) = args.array2();
    assert!(c1 == c2, "shapes not aligned");
    let s1 = squares(x1.span(), r1, c1);
    let s2 = squares(x2.span(), r2, c2);
    let mut result = array![];
    for i in 0..r1 {
        for j in 0..r2 {
            let gram = inner(x1.span().slice(i * c1, c1), x2.span().slice(j * c2, c2));
            result.append(gram.mul_int(-2) + *s1.at(i) + *s2.at(j));
        };
    };
    let mut out = args.output();
    tensor(ref out, array![r1, r2].span(), result.span());
    out
}
