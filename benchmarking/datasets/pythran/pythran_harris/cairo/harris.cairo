//! Harris response from forward differences, without smoothing.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (m, n, img): (u32, u32, Array<Q32>) = args.array2();
    let rows = if m == 0 { 0 } else { m - 1 };
    let cols = if n == 0 { 0 } else { n - 1 };
    let mut result = array![];
    for i in 0..rows {
        for j in 0..cols {
            let dx = *img.at((i + 1) * n + j + 1) - *img.at(i * n + j + 1);
            let dy = *img.at((i + 1) * n + j + 1) - *img.at((i + 1) * n + j);
            let a = dx * dx;
            let b = dy * dy;
            let c = dx * dy;
            let tr = a + b;
            result.append(a * b - c * c - (Q32 { bits: 214748365 }) * tr * tr);
        };
    };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
