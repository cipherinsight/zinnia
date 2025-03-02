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
    let mut matrix: Vec<Vec<i32>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        matrix.push(tmp);
    }
    let valid = sp1_zkvm::io::read::<i32>();

    assert!(
        (0..=1).contains(&valid),
        "Valid must be either 0 or 1, but got {}",
        valid
    );

    let n = matrix.len();

    for row in matrix.iter() {
        for &x in row.iter() {
            assert!(
                valid == 0 || (1 < x && x <= n as i32),
                "Invalid value {} in matrix when valid = {}",
                x,
                valid
            );
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
