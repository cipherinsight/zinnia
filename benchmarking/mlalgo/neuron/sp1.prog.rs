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
    let mut training_data: Vec<Vec<f64>> = Vec::new();
    for i in 0..10 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        training_data.push(tmp);
    }
    let mut training_labels: Vec<i32> = Vec::new();
    for j in 0..10 {
        training_labels.push(sp1_zkvm::io::read::<i32>());
    }
    let mut initial_weights: Vec<f64> = Vec::new();
    for j in 0..2 {
        initial_weights.push(sp1_zkvm::io::read::<f64>());
    }
    let mut testing_data: Vec<Vec<f64>> = Vec::new();
    for i in 0..2 {
        let mut tmp: Vec<f64> = Vec::new();
        for j in 0..2 {
            tmp.push(sp1_zkvm::io::read::<f64>());
        }
        testing_data.push(tmp);
    }
    let mut testing_labels: Vec<i32> = Vec::new();
    for j in 0..2 {
        testing_labels.push(sp1_zkvm::io::read::<i32>());
    }

    let n = training_data.len();

    let mut weights = [initial_weights[0], initial_weights[1]];
    // Perceptron training loop
    for _ in 0..50 {
        for i in 0..n {
            let activation = training_data[i][0] * weights[0] + training_data[i][1] * weights[1];
            let prediction = if activation >= 0.0 { 1 } else { -1 };
            if prediction != training_labels[i] {
                if training_labels[i] == 1 {
                    weights[0] += training_data[i][0];
                    weights[1] += training_data[i][1];
                } else {
                    weights[0] -= training_data[i][0];
                    weights[1] -= training_data[i][1];
                }
            }
        }
    }

    let m = testing_data.len();

    // Test the trained model
    for i in 0..m {
        let activation = testing_data[i][0] * weights[0] + testing_data[i][1] * weights[1];
        let prediction = if activation >= 0.0 { 1 } else { -1 };
        assert!(
            testing_labels[i] == (if prediction >= 0 { 1 } else { -1 }),
            "Mismatch in prediction at index {}: expected {}, but got {}",
            i,
            testing_labels[i],
            prediction
        );
    }

    // Encode the public values of the program.
    // let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { n, a, b });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    // sp1_zkvm::io::commit_slice(&bytes);
}
