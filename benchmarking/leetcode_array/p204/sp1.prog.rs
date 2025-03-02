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
    let result = sp1_zkvm::io::read::<u32>();

    assert!(n <= 1000, "n must be between 0 and 1000 inclusive");

    // For n equal to 0 or 1, the expected result is 0.
    if n == 0 || n == 1 {
        assert_eq!(result, 0, "For n = 0 or 1, result must be 0");
    } else {
        // Initialize a vector with 1001 elements set to 1.
        let mut is_prime = vec![1; 1001];
        let mut number_of_primes = 0;

        // Iterate from 2 to 1000 inclusive.
        for i in 2..=1000 {
            if is_prime[i] == 1 {
                number_of_primes += 1;
                // Mark all multiples of i as non-prime.
                for j in (i..=1000).step_by(i) {
                    is_prime[j] = 0;
                }
            }
            // When i equals n, verify that the number of primes found equals the result.
            if (i as u32) == n {
                assert_eq!(
                    number_of_primes, result,
                    "At i = {}, expected number of primes to be {}",
                    i, result
                );
            }
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
