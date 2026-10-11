//! Forward substitution with a fused inner product for each solved row.
// outputs: mutated x
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;
use crate::np::inner;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, l): (u32, u32, Array<Q32>) = args.array2();
    let x: Array<Q32> = args.array1();
    let b: Array<Q32> = args.array1();
    let n = x.len();
    let mut solved: Array<Q32> = array![];
    for i in 0..n {
        assert!(i < rows && i < cols, "L index out of bounds");
        let dot = inner(l.span().slice(i * cols, i), solved.span());
        solved.append((*b.at(i) - dot) / *l.at(i * cols + i));
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), solved.span());
    out
}
