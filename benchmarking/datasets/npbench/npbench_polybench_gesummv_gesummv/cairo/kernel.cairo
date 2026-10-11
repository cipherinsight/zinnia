//! Scale each matrix before its fused matrix-vector contraction.
// output: tensor<1,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;
use crate::np::matvec;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    let x: Array<Q32> = args.array1();
    let mut sa = array![];
    let mut sb = array![];
    for v in a { sa.append(alpha * v); };
    for v in b { sb.append(beta * v); };
    // An empty Rust row iterator never checks the vector length.
    let ax = if ar == 0 { array![] } else { matvec(sa.span(), x.span(), ar, ac) };
    let bx = if br == 0 { array![] } else { matvec(sb.span(), x.span(), br, bc) };
    // ndarray addition broadcasts either length-one operand.
    assert!(ar == br || ar == 1 || br == 1, "addition shape mismatch");
    let n = if ar == 1 { br } else { ar };
    let mut result = array![];
    for i in 0..n {
        result.append(*ax.at(if ar == 1 { 0 } else { i }) + *bx.at(if br == 1 { 0 } else { i }));
    };
    let mut out = args.output();
    tensor(ref out, array![n].span(), result.span());
    out
}
