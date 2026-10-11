//! D = ((alpha * A) @ B) @ C + beta * D, broadcasting the product to D.
// output: tensor<2,q32>
// outputs: mutated D
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    let (cr, cc, c): (u32, u32, Array<Q32>) = args.array2();
    let (dr, dc, d): (u32, u32, Array<Q32>) = args.array2();
    assert!(ac == br && bc == cr, "contraction shape mismatch");
    assert!((ar == dr || ar == 1) && (cc == dc || cc == 1), "output shape mismatch");
    let mut scaled = array![];
    for v in a {
        scaled.append(alpha * v);
    }
    let ab = crate::np::matmul(scaled.span(), b.span(), ar, ac, bc);
    let abc = crate::np::matmul(ab.span(), c.span(), ar, bc, cc);
    let mut result = array![];
    for i in 0..dr {
        for j in 0..dc {
            let pi = if ar == 1 {
                0
            } else {
                i
            };
            let pj = if cc == 1 {
                0
            } else {
                j
            };
            result.append(*abc.at(pi * cc + pj) + beta * *d.at(i * dc + j));
        };
    }
    let mut out = args.output();
    tensor(ref out, array![dr, dc].span(), result.span());
    out
}
