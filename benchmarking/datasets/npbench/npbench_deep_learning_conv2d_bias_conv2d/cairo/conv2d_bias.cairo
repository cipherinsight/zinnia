// output: tensor<4,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO};
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (n, h, w, ci, input): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (k, k2, wi, co, weights): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let bias: Array<Q32> = args.array1();
    assert!(k2 == k && wi == ci, "shape mismatch");
    assert!(h + 1 >= k && w + 1 >= k, "negative dimensions");
    assert!(bias.len() == co || bias.len() == 1, "bias shape mismatch");
    let ho = h + 1 - k;
    let wo = w + 1 - k;
    let mut result = array![];
    for b in 0..n {
        for i in 0..ho {
            for j in 0..wo {
                for channel in 0..co {
                    let mut sum = ZERO;
                    for di in 0..k {
                        for dj in 0..k {
                            for c in 0..ci {
                                let x = *input.at(((b * h + i + di) * w + j + dj) * ci + c);
                                let weight = *weights.at(((di * k + dj) * ci + c) * co + channel);
                                // Elementwise multiply followed by sum: round each product.
                                sum = sum + x * weight;
                            };
                        };
                    };
                    let bi = if bias.len() == 1 { 0 } else { channel };
                    result.append(sum + *bias.at(bi));
                };
            };
        };
    };
    let mut out = args.output();
    tensor(ref out, array![n, ho, wo, co].span(), result.span());
    out
}
