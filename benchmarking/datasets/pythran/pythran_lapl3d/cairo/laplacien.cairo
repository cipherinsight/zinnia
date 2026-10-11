//! Shape-generic Q32 port of zkvm.rs::laplacien.
// output: tensor<3,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{ONE, Q32, Q32Trait};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, depth, image): (u32, u32, u32, Array<Q32>) = args.array3();
    let nr = if rows > 2 {
        rows - 2
    } else {
        0
    };
    let nc = if cols > 2 {
        cols - 2
    } else {
        0
    };
    assert!(nr * nc * depth != 0, "empty maximum reduction");
    let mut values = array![];
    let mut valmax = ONE;
    for i in 1..nr + 1 {
        for j in 1..nc + 1 {
            for k in 0..depth {
                let pos = (i * cols + j) * depth + k;
                let v = ((*image.at(pos)).mul_int(4)
                    - *image.at(pos - cols * depth)
                    - *image.at(pos + cols * depth)
                    - *image.at(pos - depth)
                    - *image.at(pos + depth))
                    .abs();
                values.append(v);
                if v > valmax {
                    valmax = v;
                };
            };
        };
    }
    valmax = valmax + Q32 { bits: 4 };
    let mut result = array![];
    for v in values {
        result.append(v / valmax);
    }
    let mut out = args.output();
    tensor(ref out, array![nr, nc, depth].span(), result.span());
    out
}
