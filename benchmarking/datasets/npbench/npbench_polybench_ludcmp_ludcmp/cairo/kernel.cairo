//! LU factorization without pivoting, followed by forward and back substitution.
// output: (tensor<1,q32>,tensor<1,q32>)
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, ZERO, from_raw_product, raw_product};
fn get(ref a: Felt252Dict<felt252>, i: u32) -> Q32 {
    Q32 { bits: a.get(i.into()).try_into().unwrap() }
}
fn put(ref a: Felt252Dict<felt252>, i: u32, v: Q32) {
    a.insert(i.into(), v.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (n, c, values): (u32, u32, Array<Q32>) = args.array2();
    let b: Array<Q32> = args.array1();
    // Back substitution contracts all remaining columns against x.
    if n > 0 {
        assert!(c >= n && b.len() == c, "substitution shape mismatch");
    }
    let mut a: Felt252Dict<felt252> = Default::default();
    let mut x: Felt252Dict<felt252> = Default::default();
    let mut y = array![];
    for i in 0..values.len() {
        put(ref a, i, *values.at(i));
    }
    for i in 0..n {
        for j in 0..i {
            let mut raw: i128 = 0;
            for k in 0..j {
                raw += raw_product(get(ref a, i * c + k), get(ref a, k * c + j));
            }
            let v = (get(ref a, i * c + j) - from_raw_product(raw)) / get(ref a, j * c + j);
            put(ref a, i * c + j, v);
        }
        for j in i..n {
            let mut raw: i128 = 0;
            for k in 0..i {
                raw += raw_product(get(ref a, i * c + k), get(ref a, k * c + j));
            }
            let v = get(ref a, i * c + j) - from_raw_product(raw);
            put(ref a, i * c + j, v);
        };
    }
    for i in 0..n {
        let mut raw: i128 = 0;
        for k in 0..i {
            raw += raw_product(get(ref a, i * c + k), *y.at(k));
        }
        y.append(*b.at(i) - from_raw_product(raw));
    }
    for _ in n..b.len() {
        y.append(ZERO);
    }
    let mut i = n;
    while i > 0 {
        i -= 1;
        let mut raw: i128 = 0;
        for k in i + 1..c {
            raw += raw_product(get(ref a, i * c + k), get(ref x, k));
        }
        let v = (*y.at(i) - from_raw_product(raw)) / get(ref a, i * c + i);
        put(ref x, i, v);
    }
    let mut out = args.output();
    out.append(b.len().into());
    for i in 0..b.len() {
        out.append(get(ref x, i).bits);
    }
    tensor(ref out, array![b.len()].span(), y.span());
    out
}
