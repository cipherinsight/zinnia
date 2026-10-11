//! Accumulate A @ y_1 and y_2 @ A into x1 and x2, with ndarray broadcasting.
// output: (tensor<1,q32>,tensor<1,q32>)
// outputs: mutated x1, x2
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x1: Array<Q32> = args.array1();
    let x2: Array<Q32> = args.array1();
    let y1: Array<Q32> = args.array1();
    let y2: Array<Q32> = args.array1();
    let (r, c, a): (u32, u32, Array<Q32>) = args.array2();
    let p1 = crate::np::matvec(a.span(), y1.span(), r, c);
    let p2 = crate::np::vecmat(y2.span(), a.span(), r, c);
    assert!((r == x1.len() || r == 1) && (c == x2.len() || c == 1), "output shape mismatch");
    let mut result1 = array![];
    let mut result2 = array![];
    for i in 0..x1.len() {
        result1.append(*x1.at(i) + *p1.at(if r == 1 {
            0
        } else {
            i
        }));
    }
    for i in 0..x2.len() {
        result2.append(*x2.at(i) + *p2.at(if c == 1 {
            0
        } else {
            i
        }));
    }
    let mut out = args.output();
    tensor(ref out, array![x1.len()].span(), result1.span());
    tensor(ref out, array![x2.len()].span(), result2.span());
    out
}
