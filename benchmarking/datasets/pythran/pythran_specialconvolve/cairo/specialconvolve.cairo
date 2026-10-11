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
    let (rows, cols, a): (u32, u32, Array<i64>) = args.array2();
    let nr = if rows > 2 { rows - 2 } else { 0 };
    let nc = if cols > 2 { cols - 2 } else { 0 };
    let mut rowconvol: Array<i64> = array![];
    for i in 0..nr {
        for j in 0..cols {
            let middle: i128 = (*a.at((i + 1) * cols + j)).into();
            rowconvol.append(wrap_i64(middle + Into::<i64, i128>::into(*a.at(i * cols + j))
                + Into::<i64, i128>::into(*a.at((i + 2) * cols + j))));
        };
    };
    let mut result: Array<i64> = array![];
    for i in 0..nr {
        for j in 0..nc {
            let middle: i128 = (*rowconvol.at(i * cols + j + 1)).into();
            result.append(wrap_i64(middle + Into::<i64, i128>::into(*rowconvol.at(i * cols + j))
                + Into::<i64, i128>::into(*rowconvol.at(i * cols + j + 2))
                - 9 * Into::<i64, i128>::into(*a.at((i + 1) * cols + j + 1))));
        };
    };
    let mut out = args.output();
    tensor(ref out, array![nr, nc].span(), result.span());
    out
}
