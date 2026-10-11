//! Explicit-loop matrix multiplication; round each product before adding it.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO};

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r, k0, a): (u32, u32, Array<Q32>) = args.array2();
    let (k, c, b): (u32, u32, Array<Q32>) = args.array2();
    assert!(k > 0, "m1[0] of an empty list");
    // The reference allocates (c,r), not (r,c); empty loops remain valid.
    if r > 0 && c > 0 {
        assert!(r == c && k <= k0, "index out of bounds");
    }
    let mut result = array![];
    for i in 0..r {
        for j in 0..c {
            let mut sum = ZERO;
            for t in 0..k {
                sum = sum + *a.at(i * k0 + t) * *b.at(t * c + j);
            }
            result.append(sum);
        };
    }
    let mut out = args.output();
    tensor(ref out, array![c, r].span(), result.span());
    out
}
