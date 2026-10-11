//! Shape-generic Q32 port of zkvm.rs::pairwise.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, x): (u32, u32, Array<Q32>) = args.array2();
    let mut result = array![];
    for i in 0..rows {
        for j in 0..rows {
            let mut d = ZERO;
            for k in 0..cols {
                let tmp = *x.at(i * cols + k) - *x.at(j * cols + k);
                d = d + tmp * tmp;
            }
            result.append(d.sqrt());
        };
    }
    let mut out = args.output();
    tensor(ref out, array![rows, rows].span(), result.span());
    out
}
