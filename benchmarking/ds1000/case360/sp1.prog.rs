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
    let mut data: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<u64>());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<u64>> = Vec::new();
    for i in 0..(3 - 1) {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<u64>());
        }
        results.push(tmp);
    }

    for i in 0..(3 - 1) {
        for j in 0..4 {
            assert_eq!(data[i][j], results[i][j]);
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
