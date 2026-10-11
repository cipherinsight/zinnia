//! Lower-triangle rank-2k update; multiply left to right and round each product.
// outputs: mutated C
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (cr, cc, c): (u32, u32, Array<Q32>) = args.array2();
    let (n, m, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    assert!(n <= cr && n <= cc, "C slice out of bounds");
    if n > 0 && m > 0 { assert!(br >= n && bc >= m, "B index out of bounds"); };
    let mut result = array![];
    for i in 0..cr {
        for j in 0..cc {
            let mut value = *c.at(i * cc + j);
            if i < n && j <= i {
                value = value * beta;
                for k in 0..m {
                    value = value + (*a.at(j * m + k) * alpha * *b.at(i * bc + k)
                        + *b.at(j * bc + k) * alpha * *a.at(i * m + k));
                };
            };
            result.append(value);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![cr, cc].span(), result.span());
    out
}
