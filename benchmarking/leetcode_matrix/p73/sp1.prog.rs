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
    for i in 0..8 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        matrix.push(tmp);
    }
    let mut sol: Vec<Vec<i32>> = Vec::new();
    for i in 0..8 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        sol.push(tmp);
    }

    let m = 8;
    let n = 10;

    for i in 0..m {
        for j in 0..n {
            if matrix[i][j] == 0 {
                // Ensure the entire column j in sol is 0
                for row in 0..m {
                    assert_eq!(
                        sol[row][j], 0,
                        "Expected sol[{}][{}] to be 0, but got {}",
                        row, j, sol[row][j]
                    );
                }
                // Ensure the entire row i in sol is 0
                for col in 0..n {
                    assert_eq!(
                        sol[i][col], 0,
                        "Expected sol[{}][{}] to be 0, but got {}",
                        i, col, sol[i][col]
                    );
                }
            }
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
