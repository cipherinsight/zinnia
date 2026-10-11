//! Horizontal diffusion: Laplacian, limited fluxes, then divergence update.
// outputs: mutated out_field
// unused parameters: out_field
// output: tensor<3,q32>
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, ZERO};
fn at(a: Span<Q32>, cols: u32, depth: u32, i: u32, j: u32, k: u32) -> Q32 {
    *a.at((i * cols + j) * depth + k)
}
fn limit(res: Q32, grad: Q32) -> Q32 {
    if res * grad > ZERO { ZERO } else { res }
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, depth, input): (u32, u32, u32, Array<Q32>) = args.array3();
    let (ni, nj, nk, coeff): (u32, u32, u32, Array<Q32>) = args.array3();
    assert!(rows >= ni + 4 && cols >= nj + 4 && depth == nk, "hdiff shape mismatch");
    let input = input.span();
    let mut lap = array![];
    for a in 0..ni + 2 {
        for b in 0..nj + 2 {
            for k in 0..nk {
                let i = a + 1;
                let j = b + 1;
                lap.append(at(input, cols, nk, i, j, k).mul_int(4)
                    - (at(input, cols, nk, i + 1, j, k) + at(input, cols, nk, i - 1, j, k)
                    + at(input, cols, nk, i, j + 1, k) + at(input, cols, nk, i, j - 1, k)));
            };
        };
    };
    let lap = lap.span();
    let mut flx = array![];
    for a in 0..ni + 1 {
        for b in 0..nj {
            for k in 0..nk {
                let res = at(lap, nj + 2, nk, a + 1, b + 1, k) - at(lap, nj + 2, nk, a, b + 1, k);
                flx.append(limit(res, at(input, cols, nk, a + 2, b + 2, k) - at(input, cols, nk, a + 1, b + 2, k)));
            };
        };
    };
    let mut fly = array![];
    for a in 0..ni {
        for b in 0..nj + 1 {
            for k in 0..nk {
                let res = at(lap, nj + 2, nk, a + 1, b + 1, k) - at(lap, nj + 2, nk, a + 1, b, k);
                fly.append(limit(res, at(input, cols, nk, a + 2, b + 2, k) - at(input, cols, nk, a + 2, b + 1, k)));
            };
        };
    };
    let mut result = array![];
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let divergence = at(flx.span(), nj, nk, i + 1, j, k) - at(flx.span(), nj, nk, i, j, k)
                    + at(fly.span(), nj + 1, nk, i, j + 1, k) - at(fly.span(), nj + 1, nk, i, j, k);
                result.append(at(input, cols, nk, i + 2, j + 2, k) - at(coeff.span(), nj, nk, i, j, k) * divergence);
            };
        };
    };
    let mut out = args.output();
    tensor(ref out, array![ni, nj, nk].span(), result.span());
    out
}
