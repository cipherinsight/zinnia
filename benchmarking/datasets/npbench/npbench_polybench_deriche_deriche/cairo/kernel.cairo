//! Deriche row and column recurrences using the exact Rust coefficients.
// output: tensor<2,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
use crate::q32::ONE;
#[derive(Copy, Drop)]
struct Coefficients { a1: Q32, a2: Q32, a3: Q32, a4: Q32, b1: Q32, b2: Q32 }

// The backward recurrence is appended in reverse order; no mutable buffer is needed.
fn filter(x: Span<Q32>, base: u32, stride: u32, len: u32, c: Coefficients) -> Array<Q32> {
    let mut forward = array![c.a1 * *x.at(base)];
    forward.append(c.a1 * *x.at(base + stride) + c.a2 * *x.at(base) + c.b1 * *forward.at(0));
    for j in 2..len {
        forward.append(c.a1 * *x.at(base + j * stride) + c.a2 * *x.at(base + (j - 1) * stride)
            + c.b1 * *forward.at(j - 1) + c.b2 * *forward.at(j - 2));
    };
    let mut backward = array![ZERO, c.a3 * *x.at(base + (len - 1) * stride)];
    let mut j = len - 2;
    while j > 0 {
        j -= 1;
        let r = len - 1 - j;
        backward.append(c.a3 * *x.at(base + (j + 1) * stride) + c.a4 * *x.at(base + (j + 2) * stride)
            + c.b1 * *backward.at(r - 1) + c.b2 * *backward.at(r - 2));
    };
    let mut result = array![];
    for j in 0..len { result.append(*forward.at(j) + *backward.at(len - 1 - j)); };
    result
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let (rows, cols, x): (u32, u32, Array<Q32>) = args.array2();
    let two: Q32 = Q32Trait::from_int(2);
    let exp_neg = (-alpha).exp();
    let exp_neg_two = (-two * alpha).exp();
    let k = (ONE - exp_neg) * (ONE - exp_neg) / (ONE + alpha * exp_neg - (two * alpha).exp());
    let c = Coefficients { a1: k, a2: k * exp_neg * (alpha - ONE), a3: k * exp_neg * (alpha + ONE),
        a4: -k * exp_neg_two, b1: two.powf(-alpha), b2: -exp_neg_two };
    assert!(cols >= 2, "index -2 is out of bounds");
    let mut filtered = array![];
    for i in 0..rows { filtered.append_span(filter(x.span(), i * cols, 1, cols, c).span()); };
    assert!(rows >= 2, "index -2 is out of bounds");
    let mut columns: Array<Span<Q32>> = array![];
    for j in 0..cols { columns.append(filter(filtered.span(), j, cols, rows, c).span()); };
    let mut result = array![];
    for i in 0..rows { for j in 0..cols { result.append(*(*columns.at(j)).at(i)); }; };
    let mut out = args.output();
    tensor(ref out, array![rows, cols].span(), result.span());
    out
}
