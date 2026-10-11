//! Lower-triangle rank-k update, with two rounded multiplies per term.
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
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    assert!(cr >= ar && cc >= ar, "syrk slice out of bounds");
    let mut result = array![];
    for i in 0..cr {
        // The row's scale factors are reused across its lower triangle.
        let mut scales = array![];
        if i < ar { for k in 0..ac { scales.append(alpha * *a.at(i * ac + k)); }; };
        for j in 0..cc {
            let mut value = *c.at(i * cc + j);
            if i < ar && j <= i {
                value = value * beta;
                for k in 0..ac { value = value + *scales.at(k) * *a.at(j * ac + k); };
            };
            result.append(value);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![cr, cc].span(), result.span());
    out
}
