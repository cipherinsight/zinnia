//! Insertion sort with mutable storage for shifting elements.
// outputs: mutated list2
// output: tensor<1,q32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::Q32;
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let list: Array<Q32> = args.array1();
    let n = list.len();
    let mut values: Felt252Dict<felt252> = Default::default();
    for i in 0..n { values.insert(i.into(), (*list.at(i)).bits.into()); };
    for i in 1..n {
        let save: i64 = values.get(i.into()).try_into().unwrap();
        let mut j = i;
        while j > 0 {
            let previous: i64 = values.get((j - 1).into()).try_into().unwrap();
            if previous <= save { break; }
            values.insert(j.into(), previous.into());
            j -= 1;
        };
        values.insert(j.into(), save.into());
    };
    let mut out = args.output();
    out.append(n.into());
    for i in 0..n { out.append(values.get(i.into()).try_into().unwrap()); };
    out
}
