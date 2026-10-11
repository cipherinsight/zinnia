//! Pairwise distances with two sequential periodic wrapping passes per axis.
// output: (tensor<2,q32>,tensor<2,q32>,tensor<2,q32>,tensor<2,q32>)
use crate::io::{ArgsTrait, tensor};
use crate::q32::{ONE, Q32, Q32Trait, ZERO};

fn separations(v: Span<Q32>, n: u32, l: i64, periodic: bool) -> Array<Q32> {
    assert!(v.len() == n, "broadcast shape mismatch");
    let side: Q32 = Q32Trait::from_int(l);
    let upper = side.div_int(2);
    let lower = Q32Trait::from_int(-l).div_int(2);
    let mut result = array![];
    for i in 0..n {
        for j in 0..n {
            let mut s = *v.at(j) - *v.at(i);
            if periodic {
                if s > upper {
                    s = s - side;
                }
                if s < lower {
                    s = s + side;
                };
            }
            result.append(s);
        };
    }
    result
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let x: Array<Q32> = args.array1();
    let y: Array<Q32> = args.array1();
    let z: Array<Q32> = args.array1();
    let l: i64 = args.scalar();
    let px: bool = args.scalar();
    let py: bool = args.scalar();
    let pz: bool = args.scalar();
    let n = x.len();
    let dx = separations(x.span(), n, l, px);
    let dy = separations(y.span(), n, l, py);
    let dz = separations(z.span(), n, l, pz);
    let mut d = array![];
    for i in 0..n * n {
        let (a, b, c) = (*dx.at(i), *dy.at(i), *dz.at(i));
        let v = (a * a + b * b + c * c).sqrt();
        d.append(if v == ZERO {
            -ONE
        } else {
            v
        });
    }
    let mut out = args.output();
    let dims = array![n, n].span();
    tensor(ref out, dims, d.span());
    tensor(ref out, dims, dx.span());
    tensor(ref out, dims, dy.span());
    tensor(ref out, dims, dz.span());
    out
}
