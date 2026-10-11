//! Triangular product: rows below the current row still contain input B.
// outputs: mutated B
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, raw_product, from_raw_product};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (rows, cols, b): (u32, u32, Array<Q32>) = args.array2();
    if rows > 0 && cols > 0 { assert!(ar == rows && ac >= rows, "trmm shape mismatch"); };
    let mut result = array![];
    for i in 0..rows {
        for j in 0..cols {
            let mut raw: i128 = 0;
            for k in i + 1..rows { raw += raw_product(*a.at(k * ac + i), *b.at(k * cols + j)); };
            result.append((*b.at(i * cols + j) + from_raw_product(raw)) * alpha);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
