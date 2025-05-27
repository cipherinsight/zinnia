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
    for i in 0..4 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<u64>());
        }
        data.push(tmp);
    }
    let mut results: Vec<Vec<u64>> = Vec::new();
    for i in 0..4 {
        let mut tmp: Vec<u64> = Vec::new();
        for j in 0..4 {
            tmp.push(sp1_zkvm::io::read::<u64>());
        }
        results.push(tmp);
    }

    assert_eq!(data[0][0], results[0][0]);
    assert_eq!(data[0][1], results[0][1]);
    assert_eq!(data[0][2], results[1][0]);
    assert_eq!(data[0][3], results[1][1]);
    assert_eq!(data[1][0], results[0][2]);
    assert_eq!(data[1][1], results[0][3]);
    assert_eq!(data[1][2], results[1][2]);
    assert_eq!(data[1][3], results[1][3]);
    assert_eq!(data[2][0], results[2][0]);
    assert_eq!(data[2][1], results[2][1]);
    assert_eq!(data[2][2], results[3][0]);
    assert_eq!(data[2][3], results[3][1]);
    assert_eq!(data[3][0], results[2][2]);
    assert_eq!(data[3][1], results[2][3]);
    assert_eq!(data[3][2], results[3][2]);
    assert_eq!(data[3][3], results[3][3]);

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
