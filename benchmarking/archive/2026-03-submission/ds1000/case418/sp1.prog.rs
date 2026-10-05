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
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..5 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        data.push(tmp);
    }
    let mut results: Vec<f64> = Vec::new();
    for i in 0..2 {
        results.push(sp1_zkvm::io::read::<f64>());
    }

    let mut answers = vec![0.0; 2];
    for i in 0..2 {
        let mut sum = 0.0;
        let bins = 5 / 3;
        for j in (5%3)..5 {
            sum += data[i][j];
        }
        answers[i] = sum / (bins as f64) / 3.0;
    }

    assert_eq!(results[0], answers[0]);
    assert_eq!(results[1], answers[1]);

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
