//! Each k step reads the previous matrix, preserving row/column snapshots.
// outputs: mutated path
// output: tensor<2,i64>
use crate::io::{ArgsTrait, tensor};

// Rust explicitly uses wrapping_add for the path sum.
fn wrapping_add(a: i64, b: i64) -> i64 {
    let sum: i128 = Into::<i64, i128>::into(a) + b.into();
    let sum = if sum > 9223372036854775807 { sum - 18446744073709551616 }
        else if sum < -9223372036854775808 { sum + 18446744073709551616 } else { sum };
    sum.try_into().unwrap()
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, mut path): (u32, u32, Array<i64>) = args.array2();
    for k in 0..rows {
        assert!(k < cols, "column out of bounds");
        let mut next = array![];
        for i in 0..rows {
            let column = *path.at(i * cols + k);
            for j in 0..cols {
                let candidate = wrapping_add(column, *path.at(k * cols + j));
                let old = *path.at(i * cols + j);
                next.append(if candidate < old { candidate } else { old });
            };
        };
        path = next;
    };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), path.span());
    out
}
