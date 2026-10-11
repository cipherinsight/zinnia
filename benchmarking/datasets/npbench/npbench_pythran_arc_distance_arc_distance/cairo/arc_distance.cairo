//! Haversine distances, preserving the Rust Q32 evaluation order.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait};
use crate::q32::ONE;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let t1: Array<Q32> = args.array1();
    let p1: Array<Q32> = args.array1();
    let t2: Array<Q32> = args.array1();
    let p2: Array<Q32> = args.array1();
    let n = t1.len();
    assert!(p1.len() == n && t2.len() == n && p2.len() == n, "Zip shape mismatch");
    let mut result = array![];
    for i in 0..n {
        let dt = (*t2.at(i) - *t1.at(i)).div_int(2).sin();
        let dp = (*p2.at(i) - *p1.at(i)).div_int(2).sin();
        let temp = dt * dt + (*t1.at(i)).cos() * (*t2.at(i)).cos() * (dp * dp);
        result.append(temp.sqrt().atan2((ONE - temp).sqrt()).mul_int(2));
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
