// DS-1000 case 367 — Risc0 guest program.

use risc0_zkvm::guest::env;

fn main() {
    let mut pairs: Vec<Vec<i64>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let mut row: Vec<i64> = Vec::with_capacity(2);
        for _ in 0..2 {
            row.push(env::read());
        }
        pairs.push(row);
    }
    let mut array_of_arrays: Vec<i64> = Vec::with_capacity(3);
    for _ in 0..3 {
        array_of_arrays.push(env::read());
    }
    let mut claimed: Vec<i64> = Vec::with_capacity(3);
    for _ in 0..3 {
        claimed.push(env::read());
    }

    let mut result: Vec<i64> = Vec::with_capacity(3);
    for i in 0..3 {
        result.push(array_of_arrays[i]);
    }

    let _ = &pairs;

    for i in 0..3 {
        assert_eq!(result[i], claimed[i]);
    }
}
