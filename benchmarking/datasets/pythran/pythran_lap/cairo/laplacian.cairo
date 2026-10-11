//! Fourth-order Laplacian with the reference endpoint convention.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let var: Array<Q32> = args.array1();
    let dh2: Q32 = args.scalar();
    assert!(dh2 != ZERO, "requires dh2 != 0.0");
    let n = var.len();
    assert!(n >= 2, "laplacian index out of bounds");
    let near = Q32 { bits: 5726623061 };
    let center = Q32 { bits: -10737418240 };
    let far = Q32 { bits: -357913941 };
    let mut result = array![];
    for i in 0..n {
        // The reference deliberately uses var[1] at the left endpoint.
        let left = if i == 0 { 1 } else { i - 1 };
        let right = if i + 1 == n { 0 } else { i + 1 };
        let far_left = if i < 2 { n - 2 + i } else { i - 2 };
        let far_right = if i + 2 >= n { i + 2 - n } else { i + 2 };
        let value = near * *var.at(left) + near * *var.at(right) + center * *var.at(i)
            + far * *var.at(far_left) + far * *var.at(far_right);
        result.append(value / dh2);
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
