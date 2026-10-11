// unused parameters: float_n
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ONE, ZERO, raw_product, from_raw_product};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let m: i64 = args.scalar();
    let (observations, variables, data): (u32, u32, Array<Q32>) = args.array2();
    assert!(0 <= m && m <= 1024, "requires 0 <= M <= 1024");
    // Store the centered transpose contiguously, like Rust's row-wise operation.
    let mut x = array![];
    for j in 0..variables {
        let mut sum = ZERO;
        for i in 0..observations { sum = sum + *data.at(i * variables + j); };
        let mean = sum.div_int(observations.into());
        for i in 0..observations { x.append(*data.at(i * variables + j) - mean); };
    };
    let count: i64 = observations.into();
    let scale = ONE.div_int(count - 1);
    let mut result = array![];
    for i in 0..variables {
        for j in 0..variables {
            let mut raw: i128 = 0;
            for k in 0..observations {
                raw += raw_product(*x.at(i * observations + k), *x.at(j * observations + k));
            };
            result.append(from_raw_product(raw) * scale);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![variables, variables].span(), result.span());
    out
}
