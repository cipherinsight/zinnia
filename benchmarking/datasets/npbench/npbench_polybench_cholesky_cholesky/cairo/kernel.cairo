//! In-place lower-triangle Cholesky factorization with fused inner products.
// output: tensor<2,q32>
// outputs: mutated A
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::{Q32, Q32Trait, from_raw_product, raw_product};
fn get(ref a: Felt252Dict<felt252>, i: u32) -> Q32 {
    Q32 { bits: a.get(i.into()).try_into().unwrap() }
}
fn put(ref a: Felt252Dict<felt252>, i: u32, v: Q32) {
    a.insert(i.into(), v.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (r, c, values): (u32, u32, Array<Q32>) = args.array2();
    assert!(r > 0 && c >= r, "index out of bounds");
    let mut a: Felt252Dict<felt252> = Default::default();
    for i in 0..values.len() {
        put(ref a, i, *values.at(i));
    }
    let first = get(ref a, 0).sqrt();
    put(ref a, 0, first);
    for i in 1..r {
        for j in 0..i {
            let mut raw: i128 = 0;
            for k in 0..j {
                raw += raw_product(get(ref a, i * c + k), get(ref a, j * c + k));
            }
            let v = (get(ref a, i * c + j) - from_raw_product(raw)) / get(ref a, j * c + j);
            put(ref a, i * c + j, v);
        }
        let mut raw: i128 = 0;
        for k in 0..i {
            let v = get(ref a, i * c + k);
            raw += raw_product(v, v);
        }
        let v = (get(ref a, i * c + i) - from_raw_product(raw)).sqrt();
        put(ref a, i * c + i, v);
    }
    let mut out = args.output();
    out.append(r.into());
    out.append(c.into());
    for i in 0..r * c {
        out.append(get(ref a, i).bits);
    }
    out
}
