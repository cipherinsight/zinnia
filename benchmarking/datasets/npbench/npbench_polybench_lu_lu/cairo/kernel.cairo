// outputs: mutated A
// output: tensor<2,q32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::{ArgsTrait, tensor};
use crate::q32::{Q32, raw_product, from_raw_product};

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
    let (n, cols, input): (u32, u32, Array<Q32>) = args.array2();
    // Rust accesses n columns, but preserves any additional columns.
    assert!(cols >= n, "index out of bounds");
    let mut a: Felt252Dict<felt252> = Default::default();
    for i in 0..input.len() { write_q(ref a, i, *input.at(i)); };
    for i in 0..n {
        for j in 0..i {
            let mut raw: i128 = 0;
            for k in 0..j {
                raw += raw_product(read_q(ref a, i * cols + k), read_q(ref a, k * cols + j));
            };
            let pivot = read_q(ref a, j * cols + j);
            let value = (read_q(ref a, i * cols + j) - from_raw_product(raw)) / pivot;
            write_q(ref a, i * cols + j, value);
        };
        for j in i..n {
            let mut raw: i128 = 0;
            for k in 0..i {
                raw += raw_product(read_q(ref a, i * cols + k), read_q(ref a, k * cols + j));
            };
            let value = read_q(ref a, i * cols + j) - from_raw_product(raw);
            write_q(ref a, i * cols + j, value);
        };
    };
    let mut result = array![];
    for i in 0..input.len() { result.append(read_q(ref a, i)); };
    let mut out = args.output();
    tensor(ref out, array![n, cols].span(), result.span());
    out
}
