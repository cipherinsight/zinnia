//! Symmetric matrix update from the lower triangle, preserving scalar rounding.
// output: tensor<2,q32>
// outputs: mutated C
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::{Q32, from_raw_product, raw_product};
fn get(ref a: Felt252Dict<felt252>, i: u32) -> Q32 {
    Q32 { bits: a.get(i.into()).try_into().unwrap() }
}
fn put(ref a: Felt252Dict<felt252>, i: u32, v: Q32) {
    a.insert(i.into(), v.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let alpha: Q32 = args.scalar();
    let beta: Q32 = args.scalar();
    let (r, c, values): (u32, u32, Array<Q32>) = args.array2();
    let (ar, ac, a): (u32, u32, Array<Q32>) = args.array2();
    let (br, bc, b): (u32, u32, Array<Q32>) = args.array2();
    let mut result: Felt252Dict<felt252> = Default::default();
    for i in 0..values.len() {
        put(ref result, i, *values.at(i) * beta);
    }
    for i in 0..r {
        assert!(i < ar && i < ac, "A index out of bounds");
        for j in 0..c {
            assert!(i < br && j < bc, "B index out of bounds");
            let scale = alpha * *b.at(i * bc + j);
            let mut raw: i128 = 0;
            for k in 0..i {
                let av = *a.at(i * ac + k);
                let v = get(ref result, k * c + j) + scale * av;
                put(ref result, k * c + j, v);
                raw += raw_product(*b.at(k * bc + j), av);
            }
            let v = get(ref result, i * c + j)
                + (scale * *a.at(i * ac + i) + alpha * from_raw_product(raw));
            put(ref result, i * c + j, v);
        };
    }
    let mut out = args.output();
    out.append(r.into());
    out.append(c.into());
    for i in 0..r * c {
        out.append(get(ref result, i).bits);
    }
    out
}
