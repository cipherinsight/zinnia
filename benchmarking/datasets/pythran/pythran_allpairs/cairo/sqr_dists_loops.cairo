//! Squared row distances; each square is rescaled before summation.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (xr, xc, x): (u32, u32, Array<Q32>) = args.array2();
    let (yr, yc, y): (u32, u32, Array<Q32>) = args.array2();
    if xr > 0 && yr > 0 { assert!(xc == yc, "row shape mismatch"); };
    let mut result = array![];
    for a in 0..xr {
        for b in 0..yr {
            let mut sum = ZERO;
            for k in 0..xc {
                let d = *x.at(a * xc + k) - *y.at(b * yc + k);
                sum = sum + d * d;
            };
            result.append(sum);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![xr, yr].span(), result.span());
    out
}
