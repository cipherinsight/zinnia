//! Strided complex copy, retaining untouched destination elements.
// outputs: mutated CY
// output: tensor<1,c32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::C32;
use crate::np::np_index;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let n: i64 = args.scalar();
    let cx: Array<C32> = args.array1();
    let incx: i64 = args.scalar();
    let cy: Array<C32> = args.array1();
    let incy: i64 = args.scalar();
    assert!(n >= 0 && n <= 1024, "requires 0 <= N <= 1024");
    let mut values: Felt252Dict<felt252> = Default::default();
    for i in 0..cy.len() {
        let z = *cy.at(i);
        values.insert((2 * i).into(), z.re.bits.into());
        values.insert((2 * i + 1).into(), z.im.bits.into());
    };
    let mut ix = if incx < 0 { (1 - n) * incx } else { 0 };
    let mut iy = if incy < 0 { (1 - n) * incy } else { 0 };
    let count: u32 = n.try_into().unwrap();
    for _ in 0..count {
        let y = np_index(iy, cy.len());
        let z = *cx.at(np_index(ix, cx.len()));
        values.insert((2 * y).into(), z.re.bits.into());
        values.insert((2 * y + 1).into(), z.im.bits.into());
        ix += incx;
        iy += incy;
    };
    let mut out = args.output();
    out.append(cy.len().into());
    for i in 0..2 * cy.len() { out.append(values.get(i.into()).try_into().unwrap()); };
    out
}
