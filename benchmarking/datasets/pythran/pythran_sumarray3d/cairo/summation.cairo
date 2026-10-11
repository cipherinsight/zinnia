//! Weighted inverse distances and separation vectors.
// output: (tensor<1,q32>,tensor<2,q32>)
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO, ONE};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (pr, pc, pos): (u32, u32, Array<Q32>) = args.array2();
    let weights: Array<Q32> = args.array1();
    let (rows, cols, points): (u32, u32, Array<Q32>) = args.array2();
    if rows > 0 {
        assert!(cols >= 3, "point index out of bounds");
        if weights.len() > 0 { assert!(pr >= weights.len() && pc >= 3, "position index out of bounds"); };
    };
    let mut sums = array![];
    let mut vectors = array![];
    for i in 0..rows {
        let px = *points.at(i * cols);
        let py = *points.at(i * cols + 1);
        let pz = *points.at(i * cols + 2);
        let mut total = ZERO;
        let mut sx = ZERO;
        let mut sy = ZERO;
        let mut sz = ZERO;
        for j in 0..weights.len() {
            let w = *weights.at(j);
            let dx = px - *pos.at(j * pc);
            let dy = py - *pos.at(j * pc + 1);
            let dz = pz - *pos.at(j * pc + 2);
            let dr = ONE / (dx * dx + dy * dy + dz * dz).sqrt();
            total = total + w * dr;
            sx = sx + w * dx;
            sy = sy + w * dy;
            sz = sz + w * dz;
        };
        sums.append(total);
        vectors.append(sx); vectors.append(sy); vectors.append(sz);
    };
    let mut out = args.output();
    tensor(ref out, array![rows].span(), sums.span());
    tensor(ref out, array![rows, 3].span(), vectors.span());
    out
}
