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
    let mut a: Vec<Vec<Vec<u64>>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<Vec<u64>> = Vec::new();
        for j in 0..3 {
            let mut tmp2: Vec<u64> = Vec::new();
            for k in 0..2 {
                tmp2.push(sp1_zkvm::io::read::<u64>());
            }
            tmp1.push(tmp2);
        }
        a.push(tmp1);
    }
    let mut b: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..3 {
            tmp1.push(sp1_zkvm::io::read::<u64>());
        }
        b.push(tmp1);
    }
    let mut desired: Vec<Vec<u64>> = Vec::new();
    for i in 0..3 {
        let mut tmp1: Vec<u64> = Vec::new();
        for j in 0..3 {
            tmp1.push(sp1_zkvm::io::read::<u64>());
        }
        desired.push(tmp1);
    }

    for i in 0..3 {
        for j in 0..3 {
            assert!(b[i][j] == 0 || b[i][j] == 1);
            if b[i][j] == 0 {
                assert_eq!(a[i][j][0], desired[i][j]);
            } else if b[i][j] == 1 {
                assert_eq!(a[i][j][1], desired[i][j]);
            }
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
