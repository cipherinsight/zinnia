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
    let n = 20;
    let result = sp1_zkvm::io::read::<u32>();
    let mut nums = vec![0; n as usize];
    for i in 0..n {
        nums.push(sp1_zkvm::io::read::<u32>());
    }

    let mut values = vec![0; n as usize];
    // Populate values array
    for num in nums {
        values[num as usize] += num as i32;
    }

    let mut take = 0;
    let mut skip = 0;

    // Compute the maximum sum with non-adjacent selections
    for i in 0..n {
        let take_i = skip + values[i];
        let skip_i = take.max(skip);
        take = take_i;
        skip = skip_i;
    }

    assert_eq!(
        result,
        take.max(skip) as u32,
        "The computed result does not match the expected result."
    );

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
