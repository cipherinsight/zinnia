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
    let mut trust_graph: Vec<Vec<u32>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<u32> = Vec::new();
        for j in 0..10 {
            tmp.push(sp1_zkvm::io::read::<u32>());
        }
        trust_graph.push(tmp);
    }
    let judge_id = sp1_zkvm::io::read::<u32>();
    assert_eq!(trust_graph.len(), 10, "Trust graph must have 10 rows");
    for row in trust_graph.iter() {
        assert_eq!(
            row.len(),
            10,
            "Each row in the trust graph must have 10 columns"
        );
    }

    let judge_index = judge_id - 1;

    for i in 0..10 {
        for j in 0..10 {
            if i == judge_index && i != j {
                assert_eq!(
                    trust_graph[i as usize][j as usize], 0,
                    "Judge (index {}) should not trust anyone, but trusts person {}",
                    i, j
                );
            }
            if j == judge_index && i != j {
                assert_eq!(
                    trust_graph[i as usize][j as usize], 1,
                    "Everyone should trust the judge (index {}), but person {} does not",
                    j, i
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
