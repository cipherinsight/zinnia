//! Simplex projection using sorting and the last positive threshold.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
use crate::np::sort;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let v: Array<Q32> = args.array1();
    let z: i64 = args.scalar();
    let u = sort(v.span());
    let n = v.len();
    let mut cumsum = ZERO;
    let mut last: Option<(i64, Q32)> = None;
    for k in 0..n {
        let uk = *u.at(n - 1 - k);
        cumsum = cumsum + uk;
        let cssv = cumsum - Q32Trait::from_int(z);
        let ind: i64 = (k + 1).into();
        if uk - cssv.div_int(ind) > ZERO { last = Some((ind, cssv)); };
    };
    let (rho, cssv) = last.expect('cond is all false');
    let theta = cssv.div_int(rho);
    let mut result = array![];
    for vi in v {
        let w = vi - theta;
        result.append(if w > ZERO { w } else { ZERO });
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
