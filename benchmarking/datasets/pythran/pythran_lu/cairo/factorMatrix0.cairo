//! Gaussian elimination with first-maximum partial pivoting, as in zkvm.rs.
// outputs: mutated M
// output: tensor<2,q32>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::{Q32, Q32Trait};
fn get(ref m: Felt252Dict<felt252>, index: u32) -> Q32 {
    Q32 { bits: m.get(index.into()).try_into().unwrap() }
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (n, cols, values): (u32, u32, Array<Q32>) = args.array2();
    let mut m: Felt252Dict<felt252> = Default::default();
    for i in 0..values.len() { m.insert(i.into(), (*values.at(i)).bits.into()); };
    if n > 1 {
        for line in 0..n - 1 {
            assert!(line < cols, "pivot column out of bounds");
            let mut pivot_row = line;
            let mut largest = get(ref m, line * cols + line).abs();
            for i in line + 1..n {
                let magnitude = get(ref m, i * cols + line).abs();
                if magnitude > largest { largest = magnitude; pivot_row = i; };
            };
            if pivot_row != line {
                for k in 0..cols {
                    let a = m.get((line * cols + k).into());
                    let b = m.get((pivot_row * cols + k).into());
                    m.insert((line * cols + k).into(), b);
                    m.insert((pivot_row * cols + k).into(), a);
                };
            };
            let pivot = get(ref m, line * cols + line);
            for j in line + 1..n {
                let v = get(ref m, j * cols + line) / pivot;
                for k in line..cols {
                    let update = v * get(ref m, line * cols + k);
                    let next = get(ref m, j * cols + k) - update;
                    m.insert((j * cols + k).into(), next.bits.into());
                };
            };
        };
    };
    let mut out = args.output();
    out.append(n.into()); out.append(cols.into());
    for i in 0..n * cols { out.append(get(ref m, i).bits); };
    out
}
