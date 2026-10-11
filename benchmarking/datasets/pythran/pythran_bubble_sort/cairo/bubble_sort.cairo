//! Bubble sort, with every pass as in zkvm.rs (no early-exit optimization).
// output: tensor<1,i64>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;

#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let list: Array<i64> = args.array1();
    let n = list.len();
    // Actual mutable storage is needed for adjacent swaps.
    let mut values: Felt252Dict<felt252> = Default::default();
    for i in 0..n { values.insert(i.into(), (*list.at(i)).into()); };
    if n > 1 {
        for i in 0..n - 1 {
            for j in 0..n - i - 1 {
                let a: i64 = values.get(j.into()).try_into().unwrap();
                let b: i64 = values.get((j + 1).into()).try_into().unwrap();
                if a > b {
                    values.insert(j.into(), b.into());
                    values.insert((j + 1).into(), a.into());
                };
            };
        };
    };
    let mut out = args.output();
    out.append(n.into());
    for i in 0..n { out.append(values.get(i.into()).try_into().unwrap()); };
    out
}
