//! Preserve upstream's unusual first component, then multiply all of c by i.
// output: tensor<4,c32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, C32, C32Trait};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (cn, cx, cy, cz, c): (u32, u32, u32, u32, Array<C32>) = args.array4();
    let (an, ax, ay, az, a): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (bn, bx, by, bz, b): (u32, u32, u32, u32, Array<C32>) = args.array4();
    assert!(cn >= 3 && an >= 3 && bn >= 3, "component index out of bounds");
    assert!(cx == ax && cy == ay && cz == az && cx == bx && cy == by && cz == bz, "cross shape mismatch");
    let size = cx * cy * cz;
    let mut result = array![];
    for component in 0..cn {
        for p in 0..size {
            let v = if component < 3 {
                let (i, j, k, l) = if component == 0 { (0, 2, 2, 1) }
                    else if component == 1 { (2, 0, 0, 2) } else { (0, 1, 1, 0) };
                (*b.at(j * size + p)).scale(*a.at(i * size + p)) - (*b.at(l * size + p)).scale(*a.at(k * size + p))
            } else { *c.at(component * size + p) };
            result.append(C32 { re: -v.im, im: v.re });
        };
    };
    let mut out = args.output();
    tensor(ref out, array![cn, cx, cy, cz].span(), result.span());
    out
}
