//! C = (alpha * A) @ B + beta * C. Scale A before the fused contraction.
// outputs: mutated C
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;
use crate::np::matmul;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (cr, cc, c): (u32, u32, Array<Q32>) = args.array2();
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    assert!(ac == br, "gemm contraction shape mismatch");
    // ndarray zip_mut_with broadcasts the product to C.
    assert!((ar == cr || ar == 1) && (bc == cc || bc == 1), "gemm output shape mismatch");
    let mut scaled = array![];
    for v in a { scaled.append(alpha * v); };
    let product = matmul(scaled.span(), b.span(), ar, ac, bc);
    let mut result = array![];
    for i in 0..cr {
        for j in 0..cc {
            let pi = if ar == 1 { 0 } else { i };
            let pj = if bc == 1 { 0 } else { j };
            result.append(*product.at(pi * bc + pj) + beta * *c.at(i * cc + j));
        };
    };
    let mut out = args.output();
    tensor(ref out, array![cr, cc].span(), result.span());
    out
}
