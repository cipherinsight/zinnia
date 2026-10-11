//! Shape-generic Q32 port of zkvm.rs::kernel.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r, c, a): (u32, u32, Array<Q32>) = args.array2();
    let x: Array<Q32> = args.array1();
    let ax = crate::np::matvec(a.span(), x.span(), r, c);
    let result = crate::np::vecmat(ax.span(), a.span(), r, c);
    let mut out = args.output();
    tensor(ref out, array![c].span(), result.span());
    out
}
