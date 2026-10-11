//! Complex scattering self-energies, preserving zkvm.rs loop and rounding order.
// outputs: mutated Sigma
// output: tensor<5,c32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::{ArgsTrait, Word};
use crate::np::np_index;
use crate::q32::{C32, Q32, from_raw_product, raw_product};

// Rank-generic Args::array already supports these rank-five/six inputs.
// The driver currently needs its tensor-layout rank limit extended to five.
fn matrix5(dims: Span<u32>, data: Span<C32>, i: u32, j: u32, k: u32) -> Span<C32> {
    assert!(i < *dims.at(0) && j < *dims.at(1) && k < *dims.at(2), "matrix index out of bounds");
    let size = *dims.at(3) * *dims.at(4);
    data.slice(((i * *dims.at(1) + j) * *dims.at(2) + k) * size, size)
}

// Each real/imaginary contraction accumulates unscaled products, then rounds once.
fn complex_matmul(a: Span<C32>, b: Span<C32>, rows: u32, inner: u32, cols: u32) -> Array<C32> {
    assert!(a.len() == rows * inner && b.len() == inner * cols, "matmul shape mismatch");
    let mut out = array![];
    for r in 0..rows {
        for c in 0..cols {
            let mut re: i128 = 0;
            let mut im: i128 = 0;
            for k in 0..inner {
                let x = *a.at(r * inner + k);
                let y = *b.at(k * cols + c);
                re += raw_product(x.re, y.re);
                re += raw_product(-x.im, y.im);
                im += raw_product(x.re, y.im);
                im += raw_product(x.im, y.re);
            }
            out.append(C32 { re: from_raw_product(re), im: from_raw_product(im) });
        };
    }
    out
}

fn coupling(
    dims: Span<u32>, data: Span<C32>, q: u32, w: u32, a: u32, b: u32, i: u32, j: u32,
) -> C32 {
    assert!(
        q < *dims.at(0)
            && w < *dims.at(1)
            && a < *dims.at(2)
            && b < *dims.at(3)
            && i < *dims.at(4)
            && j < *dims.at(5),
        "coupling index out of bounds",
    );
    *data
        .at(
            ((((q * *dims.at(1) + w) * *dims.at(2) + a) * *dims.at(3) + b) * *dims.at(4) + i)
                * *dims.at(5)
                + j,
        )
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (na, nb, neighbours): (u32, u32, Array<i64>) = args.array2();
    let (hs, h): (Array<u32>, Array<C32>) = args.array(5);
    let (gs, g): (Array<u32>, Array<C32>) = args.array(5);
    let (ds, d): (Array<u32>, Array<C32>) = args.array(6);
    let (ss, sigma): (Array<u32>, Array<C32>) = args.array(5);
    let mut values: Felt252Dict<felt252> = Default::default();
    for p in 0..sigma.len() {
        let v = *sigma.at(p);
        values.insert((2 * p).into(), v.re.bits.into());
        values.insert((2 * p + 1).into(), v.im.bits.into());
    }
    let gr = *gs.at(3);
    let gc = *gs.at(4);
    let hr = *hs.at(3);
    let hc = *hs.at(4);
    let sr = *ss.at(3);
    let sc = *ss.at(4);
    for k in 0..*gs.at(0) {
        for e in 0..*gs.at(1) {
            for q in 0..*ds.at(0) {
                let nw = *ds.at(1);
                let end = if nw < e + 1 {
                    nw
                } else {
                    e + 1
                };
                for w in 0..end {
                    for i in 0..*ds.at(4) {
                        for j in 0..*ds.at(5) {
                            for a in 0..na {
                                for b in 0..nb {
                                    let atom = np_index(*neighbours.at(a * nb + b), *gs.at(2));
                                    let gm = matrix5(gs.span(), g.span(), k, e - w, atom);
                                    let hi = matrix5(hs.span(), h.span(), a, b, i);
                                    assert!(gc == hr, "first matmul shapes not aligned");
                                    let dhg = complex_matmul(gm, hi, gr, gc, hc);
                                    let factor = coupling(ds.span(), d.span(), q, w, a, b, i, j);
                                    let hj = matrix5(hs.span(), h.span(), a, b, j);
                                    let mut dhd = array![];
                                    for v in hj {
                                        dhd.append(*v * factor);
                                    }
                                    assert!(hc == hr, "second matmul shapes not aligned");
                                    let term = complex_matmul(dhg.span(), dhd.span(), gr, hc, hc);
                                    assert!(
                                        k < *ss.at(0) && e < *ss.at(1) && a < *ss.at(2),
                                        "Sigma index out of bounds",
                                    );
                                    // zip_mut_with broadcasts the result matrix to the Sigma slice.
                                    assert!(
                                        (gr == sr || gr == 1) && (hc == sc || hc == 1),
                                        "Sigma broadcast mismatch",
                                    );
                                    let start = ((k * *ss.at(1) + e) * *ss.at(2) + a) * sr * sc;
                                    for r in 0..sr {
                                        for c in 0..sc {
                                            let tr = if gr == 1 {
                                                0
                                            } else {
                                                r
                                            };
                                            let tc = if hc == 1 {
                                                0
                                            } else {
                                                c
                                            };
                                            let v = *term.at(tr * hc + tc);
                                            let p = 2 * (start + r * sc + c);
                                            let re: i64 = values.get(p.into()).try_into().unwrap();
                                            let im: i64 = values
                                                .get((p + 1).into())
                                                .try_into()
                                                .unwrap();
                                            values.insert(p.into(), (re + v.re.bits).into());
                                            values.insert((p + 1).into(), (im + v.im.bits).into());
                                        };
                                    };
                                };
                            };
                        };
                    };
                };
            };
        };
    }
    let mut out = args.output();
    for dim in ss {
        out.append(dim.into());
    }
    for p in 0..sigma.len() {
        Word::write(
            C32 {
                re: Q32 { bits: values.get((2 * p).into()).try_into().unwrap() },
                im: Q32 { bits: values.get((2 * p + 1).into()).try_into().unwrap() },
            },
            ref out,
        );
    }
    out
}
