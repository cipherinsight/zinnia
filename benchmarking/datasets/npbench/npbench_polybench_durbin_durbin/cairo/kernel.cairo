//! Durbin recursion with a fused reflected dot product.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ONE, raw_product, from_raw_product};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let r: Array<Q32> = args.array1();
    let mut alpha = -*r.at(0);
    let mut beta = ONE;
    let mut y = array![alpha];
    for k in 1..r.len() {
        beta = beta * (ONE - alpha * alpha);
        let mut raw: i128 = 0;
        for i in 0..k { raw += raw_product(*r.at(k - 1 - i), *y.at(i)); };
        alpha = -(*r.at(k) + from_raw_product(raw)) / beta;
        // Read the entire old prefix before replacing it.
        let mut next = array![];
        for i in 0..k { next.append(*y.at(i) + alpha * *y.at(k - 1 - i)); };
        next.append(alpha);
        y = next;
    };
    let mut out = args.output();
    tensor(ref out, array![r.len()].span(), y.span());
    out
}
