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
    let area = sp1_zkvm::io::read::<i32>();
    let expected_l = sp1_zkvm::io::read::<i32>();
    let expected_w = sp1_zkvm::io::read::<i32>();

    let mut w = area;

    for i in 1..=1000 {
        if area % i == 0 {
            w = i;
        }
        if i * i >= area {
            break;
        }
    }

    let mut answer_l = area / w;
    let mut answer_w = w;

    if answer_w > answer_l {
        std::mem::swap(&mut answer_l, &mut answer_w);
    }

    assert!(
        answer_l == expected_l && answer_w == expected_w,
        "Expected dimensions ({}, {}), but got ({}, {})",
        expected_l,
        expected_w,
        answer_l,
        answer_w
    );

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
