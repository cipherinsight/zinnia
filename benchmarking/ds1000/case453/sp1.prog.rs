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
    // Read an input to the program.
    //
    // Behind the scenes, this compiles down to a custom system call which handles reading inputs
    // from the prover.
    let mut data: Vec<Vec<f64>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<f64>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        results.push(tmp);
    }

    let mut a = data.clone();
    for i in 0..5 {
        for j in 0..4 {
            let p = (data[i][j] * data[i][j]);
            a[i][j] = p;
        }
    }

    let mut sum_each_row = vec![0.0; 5];
    for i in 0..5 {
        let mut tmp = 0.0;
        for j in 0..4 {
            tmp += a[i][j];
        }
        sum_each_row[i] = tmp.sqrt();
    }

    for i in 0..5 {
        for j in 0..4 {
            assert_eq!(results[i][j], data[i][j] / sum_each_row[i]);
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
