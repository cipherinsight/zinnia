//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::PublicValuesStruct;

pub fn main() {
    let rows: usize = sp1_zkvm::io::read::<i32>() as usize;
    let cols: usize = sp1_zkvm::io::read::<i32>() as usize;
    assert!(rows > 0);
    assert!(cols > 0);

    let mut data: Vec<i32> = Vec::new();
    for _ in 0..rows {
        data.push(sp1_zkvm::io::read::<i32>());
    }

    let mut result: Vec<Vec<i32>> = Vec::new();
    for _ in 0..rows {
        let mut tmp: Vec<i32> = Vec::new();
        for _ in 0..cols {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        result.push(tmp);
    }

    let mut max_value = data[0];
    for v in &data {
        if *v > max_value {
            max_value = *v;
        }
    }
    assert_eq!(cols, (max_value + 1) as usize);

    for row in &result {
        assert_eq!(row.len(), cols);
    }

    for i in 0..rows {
        for j in 0..cols {
            if (j as i32) == data[i] {
                assert_eq!(result[i][j], 1);
            } else {
                assert_eq!(result[i][j], 0);
            }
        }
    }
}
