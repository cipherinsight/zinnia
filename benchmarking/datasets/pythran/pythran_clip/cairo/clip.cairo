//! Clip complex magnitudes, preserving the epsilon branch and division order.
// output: tensor<1,c32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, C32, C32Trait, C_ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let z: Array<C32> = args.array1();
    let max: Q32 = args.scalar();
    let n = z.len();
    let mut result = array![];
    for v in z {
        let magnitude = v.abs();
        result.append(if magnitude > max {
            let unit = if magnitude < (Q32 { bits: 4295 }) { C_ZERO } else { v.div_real(magnitude) };
            unit.scale(max)
        } else { v });
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
