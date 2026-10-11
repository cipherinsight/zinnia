//! Shape-generic Q32 port of zkvm.rs::goodExpoMeasure.
// output: tensor<2,i64>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
fn exposure(value: i64, sigma: Q32) -> Q32 {
    let d = Q32Trait::from_int(value) - Q32Trait::from_int(128);
    (-(d * d) / sigma).exp()
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (channels, rows, cols, rgb): (u32, u32, u32, Array<i64>) = args.array3();
    let sigma: Q32 = args.scalar();
    assert!(sigma != ZERO, "requires sigma != 0");
    assert!(channels >= 3, "channel out of bounds");
    let size = rows * cols;
    let mut result: Array<i64> = array![];
    for p in 0..size {
        let r = exposure(*rgb.at(p), sigma);
        let g = exposure(*rgb.at(size + p), sigma);
        let b = exposure(*rgb.at(2 * size + p), sigma);
        let bits = (r * g * b).mul_int(100).bits;
        let floor = crate::q32::floor_div(bits.into(), 4294967296);
        let frac = Into::<i64, i128>::into(bits) - floor * 4294967296;
        let k = if frac > 2147483648 || (frac == 2147483648 && floor % 2 != 0) {
            floor + 1
        } else {
            floor
        };
        result.append((k * 255 / 100).try_into().unwrap());
    }
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
