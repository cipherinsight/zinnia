//! Left-associated fused matrix products: ((A @ B) @ C) @ D.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::Q32;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    let (cr, cc, c): (u32, u32, Array<Q32>) = args.array2();
    let (dr, dc, d): (u32, u32, Array<Q32>) = args.array2();
    assert!(ac == br && bc == cr && cc == dr, "contraction shape mismatch");
    let ab = crate::np::matmul(a.span(), b.span(), ar, ac, bc);
    let abc = crate::np::matmul(ab.span(), c.span(), ar, bc, cc);
    let result = crate::np::matmul(abc.span(), d.span(), ar, cc, dc);
    let mut out = args.output();
    tensor(ref out, array![ar, dc].span(), result.span());
    out
}
