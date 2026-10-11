//! Three dense layers with fused contractions, ReLU, then row softmax.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
use crate::np::matmul;
fn dense(x: Span<Q32>, w: Span<Q32>, b: Span<Q32>, rows: u32, inner: u32, cols: u32, relu: bool) -> Array<Q32> {
    assert!(b.len() == cols, "bias shape mismatch");
    let product = matmul(x, w, rows, inner, cols);
    let mut result = array![];
    for i in 0..rows {
        for j in 0..cols {
            let v = *product.at(i * cols + j) + *b.at(j);
            result.append(if relu && v < ZERO { ZERO } else { v });
        };
    };
    result
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, input): (u32, u32, Array<Q32>) = args.array2();
    let (r1, c1, w1): (u32, u32, Array<Q32>) = args.array2();
    let b1: Array<Q32> = args.array1();
    let (r2, c2, w2): (u32, u32, Array<Q32>) = args.array2();
    let b2: Array<Q32> = args.array1();
    let (r3, c3, w3): (u32, u32, Array<Q32>) = args.array2();
    let b3: Array<Q32> = args.array1();
    assert!(cols == r1 && c1 == r2 && c2 == r3, "matmul shape mismatch");
    let x = dense(input.span(), w1.span(), b1.span(), rows, cols, c1, true);
    let x = dense(x.span(), w2.span(), b2.span(), rows, c1, c2, true);
    let x = dense(x.span(), w3.span(), b3.span(), rows, c2, c3, false);
    let result = normalize(x.span(), rows, c3);
    let mut out = args.output();
    tensor(ref out, array![rows, c3].span(), result.span());
    out
}

// Normalize each contiguous lane, preserving the exp/sum/divide order.
fn normalize(x: Span<Q32>, rows: u32, cols: u32) -> Array<Q32> {
    let mut result = array![];
    for i in 0..rows {
        assert!(cols > 0, "maximum of an empty lane");
        let lane = x.slice(i * cols, cols);
        let mut maximum = *lane.at(0);
        for v in lane { if *v > maximum { maximum = *v; }; };
        let mut exps = array![];
        let mut sum = ZERO;
        for v in lane {
            let e = (*v - maximum).exp();
            exps.append(e);
            sum = sum + e;
        };
        for e in exps { result.append(e / sum); };
    };
    result
}
