// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ONE};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let theta1: Array<Q32> = args.array1();
    let phi1: Array<Q32> = args.array1();
    let theta2: Array<Q32> = args.array1();
    let phi2: Array<Q32> = args.array1();
    let n = theta1.len();
    assert!(phi1.len() == n && theta2.len() == n && phi2.len() == n, "shape mismatch");
    let mut result = array![];
    for i in 0..n {
        let t1 = *theta1.at(i);
        let t2 = *theta2.at(i);
        let dtheta = ((t2 - t1).div_int(2)).sin();
        let dphi = ((*phi2.at(i) - *phi1.at(i)).div_int(2)).sin();
        let temp = dtheta * dtheta + t1.cos() * t2.cos() * (dphi * dphi);
        result.append(temp.sqrt().atan2((ONE - temp).sqrt()).mul_int(2));
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
