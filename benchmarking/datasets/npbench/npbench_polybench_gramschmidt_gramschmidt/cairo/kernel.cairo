// output: (tensor<2,q32>,tensor<2,q32>)
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, Q32Trait, raw_product, from_raw_product};

// Cairo arrays are immutable; store Q32's signed bits for in-place updates.
fn read_q(ref values: Felt252Dict<felt252>, index: u32) -> Q32 {
    Q32 { bits: values.get(index.into()).try_into().unwrap() }
}
fn write_q(ref values: Felt252Dict<felt252>, index: u32, value: Q32) {
    values.insert(index.into(), value.bits.into());
}

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, n, input): (u32, u32, Array<Q32>) = args.array2();
    let mut a: Felt252Dict<felt252> = Default::default();
    let mut q: Felt252Dict<felt252> = Default::default();
    let mut r: Felt252Dict<felt252> = Default::default();
    for i in 0..input.len() { write_q(ref a, i, *input.at(i)); };
    for k in 0..n {
        let mut raw: i128 = 0;
        for i in 0..rows {
            let value = read_q(ref a, i * n + k);
            raw += raw_product(value, value);
        };
        let rkk = from_raw_product(raw).sqrt();
        write_q(ref r, k * n + k, rkk);
        for i in 0..rows {
            let value = read_q(ref a, i * n + k) / rkk;
            write_q(ref q, i * n + k, value);
        };
        for j in k + 1..n {
            let mut raw: i128 = 0;
            for i in 0..rows {
                raw += raw_product(read_q(ref q, i * n + k), read_q(ref a, i * n + j));
            };
            let rkj = from_raw_product(raw);
            write_q(ref r, k * n + j, rkj);
            for i in 0..rows {
                let value = read_q(ref a, i * n + j) - read_q(ref q, i * n + k) * rkj;
                write_q(ref a, i * n + j, value);
            };
        };
    };
    let mut q_values = array![];
    let mut r_values = array![];
    for i in 0..rows * n { q_values.append(read_q(ref q, i)); };
    // Unwritten lower-triangular dictionary entries default to zero.
    for i in 0..n * n { r_values.append(read_q(ref r, i)); };
    let mut out = args.output();
    tensor(ref out, array![rows, n].span(), q_values.span());
    tensor(ref out, array![n, n].span(), r_values.span());
    out
}
