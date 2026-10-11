//! Filter-bank correlation accumulated into output; every scalar product is rescaled.
// output: tensor<4,q32>
// outputs: mutated output
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::Q32;
fn get(ref a: Felt252Dict<felt252>, i: u32) -> Q32 {
    Q32 { bits: a.get(i.into()).try_into().unwrap() }
}
fn put(ref a: Felt252Dict<felt252>, i: u32, v: Q32) {
    a.insert(i.into(), v.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (ni, nr, nc, ch, imgs): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (nf, h, w, fc, filters): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let (on, out_filters, orr, oc, values): (u32, u32, u32, u32, Array<Q32>) = args.array4();
    let rows = if h <= nr + 1 {
        nr + 1 - h
    } else {
        0
    };
    let cols = if w <= nc + 1 {
        nc + 1 - w
    } else {
        0
    };
    let mut output: Felt252Dict<felt252> = Default::default();
    for i in 0..values.len() {
        put(ref output, i, *values.at(i));
    }
    for ii in 0..ni {
        for rr in 0..rows {
            for cc in 0..cols {
                for hh in 0..h {
                    for ww in 0..w {
                        for jj in 0..ch {
                            let imgval = *imgs.at(((ii * nr + rr + hh) * nc + cc + ww) * ch + jj);
                            for ff in 0..nf {
                                assert!(
                                    jj < fc && ii < on && ff < out_filters && rr < orr && cc < oc,
                                    "index out out_filters bounds",
                                );
                                let idx = ((ii * out_filters + ff) * orr + rr) * oc + cc;
                                let v = get(ref output, idx)
                                    + imgval * *filters.at(((ff * h + hh) * w + ww) * fc + jj);
                                put(ref output, idx, v);
                            };
                        };
                    };
                };
            };
        };
    }
    let mut out = args.output();
    out.append(on.into());
    out.append(out_filters.into());
    out.append(orr.into());
    out.append(oc.into());
    for i in 0..values.len() {
        out.append(get(ref output, i).bits);
    }
    out
}
