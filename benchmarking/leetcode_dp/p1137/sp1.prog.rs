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
    let n = sp1_zkvm::io::read::<u32>();
    let sol = sp1_zkvm::io::read::<u128>();
    assert!(n != 0 || sol == 0, "For n = 0, sol must be 0");
    assert!(n != 1 || sol == 1, "For n = 1, sol must be 1");
    assert!(n != 2 || sol == 1, "For n = 2, sol must be 1");

    let (mut a, mut b, mut c) = (0, 1, 1);

    for i in 3..=100 {
        let next = a + b + c;
        a = b;
        b = c;
        c = next;

        assert!(
            n != i || sol == c,
            "For n = {}, expected sol = {}, but got {}",
            i,
            c,
            sol
        );
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
