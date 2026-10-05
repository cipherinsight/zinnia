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
    let m = 5;
    let n = 5;
    let mut res = 0;

    let mut bank: Vec<Vec<i32>> = Vec::new();
    for i in 0..5 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..5 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        bank.push(tmp);
    }
    let expected = sp1_zkvm::io::read::<i32>();

    for si in 0..n {
        for sj in 0..m {
            for ti in 0..n {
                for tj in 0..m {
                    let mut add_one = bank[si][sj] == 1 && bank[ti][tj] == 1 && si < ti;

                    for k in (si + 1)..ti {
                        if (sj..tj).any(|j| bank[k][j] == 1) {
                            add_one = false;
                            break;
                        }
                    }

                    if add_one {
                        res += 1;
                    }
                }
            }
        }
    }

    assert!(res == expected, "Expected {}, but got {}", expected, res);

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
