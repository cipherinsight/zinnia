//! Shape-generic Q32 port of zkvm.rs::ising_step.
// output: tensor<2,i64>
use core::dict::{Felt252Dict, Felt252DictTrait};
use crate::io::ArgsTrait;
use crate::q32::{Q32, Q32Trait};
fn neighbour(n: u32, offset: u32, len: u32) -> u32 {
    if offset == 0 {
        if n == 0 {
            len - 1
        } else {
            n - 1
        }
    } else if offset == 2 {
        if n + 1 == len {
            0
        } else {
            n + 1
        }
    } else {
        n
    }
}
#[executable]
fn main(inputs: Array<Array<i64>>) -> Array<i64> {
    let mut args = ArgsTrait::new(inputs);
    let (rows, cols, field): (u32, u32, Array<i64>) = args.array2();
    let beta: Q32 = args.scalar();
    let mut spins: Felt252Dict<felt252> = Default::default();
    for p in 0..field.len() {
        spins.insert(p.into(), (*field.at(p)).into());
    }
    for no in 0..2_u32 {
        for mo in 0..2_u32 {
            let mut n = no;
            while n < rows {
                let mut m = mo;
                while m < cols {
                    let mut total: i64 = 0;
                    for di in 0..3_u32 {
                        for dj in 0..3_u32 {
                            if di != 1 || dj != 1 {
                                let i = neighbour(n, di, rows);
                                let j = neighbour(m, dj, cols);
                                let v: i64 = spins.get((i * cols + j).into()).try_into().unwrap();
                                total += v;
                            };
                        };
                    }
                    let p = n * cols + m;
                    let v: i64 = spins.get(p.into()).try_into().unwrap();
                    let de = 2 * v * total;
                    if de <= 0 || beta.mul_int(-de).exp() > crate::q32::HALF {
                        spins.insert(p.into(), (-v).into());
                    }
                    m += 2;
                }
                n += 2;
            };
        };
    }
    let mut out = args.output();
    out.append(rows.into());
    out.append(cols.into());
    for p in 0..field.len() {
        out.append(spins.get(p.into()).try_into().unwrap());
    }
    out
}
