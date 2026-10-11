// output: tensor<2,i64>
use crate::io::{ArgsTrait, tensor};

// Rust release int64 arithmetic retains the low 64 bits.
fn wrap_i64(value: i128) -> i64 {
    let rem = value % 18446744073709551616;
    let signed = if rem > 9223372036854775807 { rem - 18446744073709551616 }
        else if rem < -9223372036854775808 { rem + 18446744073709551616 }
        else { rem };
    signed.try_into().unwrap()
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r, c, x): (u32, u32, Array<i64>) = args.array2();
    let (yr, yc, y): (u32, u32, Array<i64>) = args.array2();
    let a: i64 = args.scalar();
    let b: i64 = args.scalar();
    let offset: i64 = args.scalar();
    assert!(r == yr && c == yc, "shape mismatch");
    let mut result = array![];
    for i in 0..x.len() {
        let v = *x.at(i);
        let clipped: i128 = (if v < 2 { 2_i64 } else if v > 10 { 10_i64 } else { v }).into();
        let yi: i128 = (*y.at(i)).into();
        result.append(wrap_i64(clipped * a.into() + yi * b.into() + offset.into()));
    };
    let mut out = args.output();
    tensor(ref out, array![r, c].span(), result.span());
    out
}
