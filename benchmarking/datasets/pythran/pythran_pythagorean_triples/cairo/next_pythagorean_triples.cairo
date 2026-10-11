//! Berggren matrices, with the Rust release-mode i64 arithmetic.
// output: tensor<2,i64>
use crate::io::{ArgsTrait, tensor};
const MATRICES: [i64; 27] = [-1,2,2, -2,1,2, -2,2,3, 1,2,2, 2,1,2, 2,2,3, 1,-2,2, 2,-1,2, 2,-2,3];
fn wrap(value: i128) -> i64 {
    let modulus: i128 = 18446744073709551616;
    let rem = value % modulus;
    let rem = if rem > 9223372036854775807 { rem - modulus }
        else if rem < -9223372036854775808 { rem + modulus } else { rem };
    rem.try_into().unwrap()
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, previous): (u32, u32, Array<i64>) = args.array2();
    assert!(cols == 3, "shapes not aligned");
    let matrices = MATRICES.span();
    let mut result = array![];
    for i in 0..rows {
        for r in 0..9_u32 {
            let mut sum: i128 = 0;
            for c in 0..3_u32 {
                let coefficient: i128 = (*matrices.at(r * 3 + c)).into();
                sum += coefficient * Into::<i64, i128>::into(*previous.at(i * 3 + c));
            };
            result.append(wrap(sum));
        };
    };
    let mut out = args.output();
    tensor(ref out, array![3 * rows, 3].span(), result.span());
    out
}
