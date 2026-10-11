//! Shape-generic Q32 port of zkvm.rs::weights.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{ONE, Q32, Q32Trait};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, data): (u32, u32, Array<i64>) = args.array2();
    let threshold: Q32 = args.scalar();
    let bound = threshold.mul_int(cols.into());
    let mut result = array![];
    for i in 0..rows {
        let mut total: i64 = 0;
        for j in 0..rows {
            let mut count: i64 = 0;
            for k in 0..cols {
                if *data.at(i * cols + k) == *data.at(j * cols + k) {
                    count += 1;
                };
            }
            if Q32Trait::from_int(count) > bound {
                total += 1;
            };
        }
        result.append(ONE.div_int(total));
    }
    let mut out = args.output();
    tensor(ref out, array![rows].span(), result.span());
    out
}
