//! BiCG: two fused matrix-vector contractions.
// output: (tensor<1,q32>,tensor<1,q32>)
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;
use crate::np::{vecmat, matvec};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, a): (u32, u32, Array<Q32>) = args.array2();
    let p: Array<Q32> = args.array1();
    let r: Array<Q32> = args.array1();
    // Rust only checks contraction lengths when an output element is computed.
    let s = if cols == 0 { array![] } else { vecmat(r.span(), a.span(), rows, cols) };
    let q = if rows == 0 { array![] } else { matvec(a.span(), p.span(), rows, cols) };
    let mut out = args.output();
    tensor(ref out, array![cols].span(), s.span());
    tensor(ref out, array![rows].span(), q.span());
    out
}
