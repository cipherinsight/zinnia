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
    let mut graph: Vec<Vec<i32>> = Vec::new();
    let mut disappear: Vec<i32> = Vec::new();
    let mut answers: Vec<i32> = Vec::new();

    for i in 0..10 {
        let mut tmp: Vec<i32> = Vec::new();
        for j in 0..10 {
            tmp.push(sp1_zkvm::io::read::<i32>());
        }
        graph.push(tmp);
    }
    for i in 0..10 {
        disappear.push(sp1_zkvm::io::read::<i32>());
    }
    for i in 0..10 {
        answers.push(sp1_zkvm::io::read::<i32>());
    }

    let n = 10;
    // Floyd-Warshall algorithm for shortest paths
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if graph[i][k] != -1 && graph[k][j] != -1 {
                    graph[i][j] = graph[i][j].min(graph[i][k] + graph[k][j]);
                }
            }
        }
    }

    // Validate the answers based on graph distances and disappear times
    for i in 0..n {
        if graph[0][i] != -1 {
            assert_eq!(
                answers[i], -1,
                "Expected answers[{}] to be -1, but got {}",
                i, answers[i]
            );
        } else if disappear[i] <= graph[0][i] {
            assert_eq!(
                answers[i], graph[0][i],
                "Expected answers[{}] to be {}, but got {}",
                i, graph[0][i], answers[i]
            );
        } else {
            assert_eq!(
                answers[i], -1,
                "Expected answers[{}] to be -1, but got {}",
                i, answers[i]
            );
        }
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
