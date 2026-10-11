//! Preserve the upstream clamp to kernel height, including on the column axis.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO};

fn clamp(i: u32, offset: u32, height: u32) -> u32 {
    let value: i64 = Into::<u32, i64>::into(i) + offset.into() - (height / 2).into();
    if value < 0 { 0 } else if value > height.into() { height } else { value.try_into().unwrap() }
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, x): (u32, u32, Array<Q32>) = args.array2();
    let (kh, kw, weights): (u32, u32, Array<Q32>) = args.array2();
    let mut result = array![];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = ZERO;
            for ii in 0..kh {
                for jj in 0..kw {
                    let r = clamp(i, ii, kh);
                    let c = clamp(j, jj, kh);
                    assert!(r < rows && c < cols, "x index out of bounds");
                    acc = acc + *x.at(r * cols + c) * *weights.at(ii * kw + jj);
                };
            };
            result.append(acc);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
